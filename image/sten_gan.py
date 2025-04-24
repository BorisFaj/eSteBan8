import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F
from decoder import Decoder
from encoder import Encoder
from discriminator import Discriminator
from torch import amp
import math
from style_loss import StyleLossHelper
from dotenv import load_dotenv
import os
from data_handler import DataHandler

load_dotenv()

torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

scaler = amp.GradScaler()

WARM_UP_LEN = int(os.getenv("WARM_UP_LEN"))  # numero de epochs que dejo al discriminador sin entrenar
image_loss_lambda = float(os.getenv("image_loss_lambda"))  # Parametro para darle algo de tolerancia al image loss
DISC_FREEZE_WINDOW = int(os.getenv("DISC_FREEZE_WINDOW"))  # ventana MAXIMA de epochs que se queda sin entrenar el discriminador despues del warmup
FREEZE_DISC_LOSS = float(os.getenv("FREEZE_DISC_LOSS"))  # loss maximo que alcanza el discriminador antes de ser congelado

image_channels = int(os.getenv("image_channels"))
image_size = int(os.getenv("image_size"))
message_size = int(os.getenv("message_size"))  # Aumentar a 512
batch_size = int(os.getenv("batch_size"))
num_epochs = int(os.getenv("num_epochs"))
IMAGE_INPUT_RES = int(os.getenv("IMAGE_INPUT_RES"))  # resolucion de la imagen de entrada
EPOCHS_TO_VAL = int(os.getenv("EPOCHS_TO_VAL"))  # numero de epochs entre validaciones
EPOCHS_TO_SAVE = int(os.getenv("EPOCHS_TO_SAVE"))  # numero de epochs para guardar el modelo
noise_std = float(os.getenv("noise_std"))  # ruido que se le mete a la imagen generada. Entre 0.01 y 0.05 es razonable para imágenes normalizadas
RUN_NAME = os.getenv("RUN_NAME")
log_dir = os.getenv("log_dir")
checkpoint_dir = os.getenv("checkpoint_dir")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

def freeze_disc(global_step: int, epoch: int, k: float=0.005) -> int:

    freeze_window = max(1, int(DISC_FREEZE_WINDOW * math.exp(-k * epoch)))
    return global_step % freeze_window == 0


def evaluate_on_testset(encoder, decoder, discriminator, test_loader, writer, device, epoch):
    encoder.eval()
    decoder.eval()
    discriminator.eval()

    total_message_loss = 0
    total_image_loss = 0
    total_bit_accuracy = 0
    total_style_loss = 0
    num_batches = 0

    bce = nn.BCEWithLogitsLoss()
    style_loss_helper = StyleLossHelper(device)

    with torch.no_grad():
        for i, (images, messages) in enumerate(test_loader):
            images = images.to(device)
            messages = messages.to(device)

            # Forward
            stego_images = encoder(images, messages)
            recovered_messages = decoder(stego_images)

            # Denormalizar imágenes para SSIM (de [-1,1] → [0,1])
            images_01 = (images + 1) / 2
            stego_images_01 = (stego_images + 1) / 2

            # Métricas
            message_loss = bce(recovered_messages, messages)
            image_loss = (1 - ssim(stego_images_01, images_01, data_range=1.0, size_average=True)) + \
                         image_loss_lambda * F.mse_loss(stego_images, images)

            pred_bits = (torch.sigmoid(recovered_messages) > 0.5).int()
            true_bits = messages.int().to(pred_bits.device)
            bit_accuracy = (pred_bits == true_bits).float().mean()

            total_message_loss += message_loss.item()
            total_image_loss += image_loss.detach().item()
            total_bit_accuracy += bit_accuracy.item()
            total_style_loss += style_loss_helper(images_01, stego_images_01).item()  # Ojo! Esto chupa!


            num_batches += 1

        avg_message_loss = total_message_loss / num_batches
        avg_image_loss = total_image_loss / num_batches
        avg_bit_accuracy = total_bit_accuracy / num_batches
        avg_style_loss = total_style_loss / num_batches

        writer.add_scalar("Test/Loss/Message", avg_message_loss, epoch)
        writer.add_scalar("Test/Loss/Image", avg_image_loss, epoch)
        writer.add_scalar("Test/Accuracy/Bit", avg_bit_accuracy, epoch)
        writer.add_scalar("Test/Style/Loss", avg_style_loss, epoch)

        # Imágenes ejemplo
        img_grid_real = make_grid(images_01[:8].cpu(), nrow=4, normalize=True)
        img_grid_stego = make_grid(stego_images_01[:8].cpu(), nrow=4, normalize=True)
        writer.add_image("Test/Images/Real", img_grid_real, epoch)
        writer.add_image("Test/Images/Stego", img_grid_stego, epoch)

    encoder.train()
    decoder.train()
    discriminator.train()


def load_latest_checkpoint(checkpoint_dir: str, run_name: str, encoder, decoder, discriminator):
    _check_point_dir = os.path.join(checkpoint_dir, run_name)

    if os.path.exists(_check_point_dir):
        checkpoints = [f for f in os.listdir(_check_point_dir) if f.endswith(".pt") and "encoder" in f]
        if not checkpoints:
            return 0  # No checkpoint found, start from epoch 0
    else:
        return 0

    # Extraer el número de epoch del nombre de archivo
    get_epoch = lambda f: int(f.split("_epoch")[1].split(".pt")[0])
    latest_epoch = max(get_epoch(f) for f in checkpoints)

    print(f"🔁 Cargando checkpoint del epoch {latest_epoch}")

    encoder.load_state_dict(torch.load(os.path.join(_check_point_dir, f"encoder_epoch{latest_epoch}.pt")))
    decoder.load_state_dict(torch.load(os.path.join(_check_point_dir, f"decoder_epoch{latest_epoch}.pt")), strict=False)
    discriminator.load_state_dict(torch.load(os.path.join(_check_point_dir, f"discriminator_epoch{latest_epoch}.pt")))

    return latest_epoch

def get_noisy(image):
    if noise_std > 0:
        noise = torch.randn_like(image) * noise_std
        _image = image + noise
        _image = torch.clamp(_image, -1, 1)

        return _image
    else:
        return image

# Entrenamiento
writer = SummaryWriter(log_dir)

train_dataset, train_loader, test_dataset, test_loader = DataHandler(batch_size=batch_size).get()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Fuerza float16
encoder = Encoder(image_channels=image_channels, message_size=message_size, image_size=image_size).half().to(device)
decoder = Decoder(image_channels=image_channels, message_size=message_size).half().to(device)
discriminator = Discriminator(image_channels=image_channels).half().to(device)

enc_dec_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

bce = nn.BCEWithLogitsLoss()

global_step = 0
start_epoch = load_latest_checkpoint(checkpoint_dir, RUN_NAME, encoder, decoder, discriminator)
for epoch in range(start_epoch, num_epochs):
    total_image_loss = 0
    total_message_loss = 0
    total_disc_loss = 0
    total_adv_loss = 0
    num_batches = 0
    total_bit_accuracy = 0
    train_discriminator = False

    for i, (images, messages) in enumerate(train_loader):
        images = get_noisy(images).half().to(device)
        messages = messages.half().to(device)

        if noise_std > 0:
            noise = torch.randn_like(images) * noise_std
            images = images + noise
            images = torch.clamp(images, -1, 1)  # mantén en el rango [-1, 1]

        # Paso forward
        with amp.autocast("cuda"):
            stego_images = encoder(images, messages)
            recovered_messages = decoder(stego_images)

            # El discriminador NO se entrena todos los epochs
            if train_discriminator:
                # Se sigue calculando la perdida dentro del grafo
                disc_real = discriminator(images)
                disc_fake = discriminator(stego_images.detach())

                disc_loss = F.mse_loss(disc_real, torch.ones_like(disc_real)) + \
                            F.mse_loss(disc_fake, torch.zeros_like(disc_fake))

                disc_opt.zero_grad()
                scaler.scale(disc_loss).backward()
                scaler.step(disc_opt)
                scaler.update()

                if disc_loss.item() <= FREEZE_DISC_LOSS:
                    train_discriminator = False  # Deja de entrenar

            else: # si no esta entrenando
                 if epoch > WARM_UP_LEN and freeze_disc(global_step, epoch):
                     train_discriminator = True  # Empieza a entrenar

            # Encoder + Decoder
            disc_pred = discriminator(stego_images)
            image_loss = (1 - ssim(stego_images, images, data_range=1.0, size_average=True)) + \
                         image_loss_lambda * F.mse_loss(stego_images, images)

            message_loss = bce(recovered_messages, messages)

            # WarmUP
            if epoch < WARM_UP_LEN or not train_discriminator:
                total_loss = message_loss # Si no entrena discriminador, tampoco entra en el total_loss
            else:
                adv_loss = F.mse_loss(disc_pred, torch.ones_like(disc_pred))
                total_loss = message_loss + adv_loss

        with torch.no_grad():
            # Bit Accuracy
            pred_bits = (torch.sigmoid(recovered_messages) > 0.5).int()
            true_bits = messages.int().to(pred_bits.device)
            bit_accuracy = (pred_bits == true_bits).float().mean()

            if not train_discriminator:
                adv_loss = F.mse_loss(disc_pred, torch.ones_like(disc_pred))
                disc_real = discriminator(images)
                disc_fake = discriminator(stego_images.detach())

                disc_loss = F.mse_loss(disc_real, torch.ones_like(disc_real)) + \
                            F.mse_loss(disc_fake, torch.zeros_like(disc_fake))

        enc_dec_opt.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(enc_dec_opt)
        scaler.update()

        del total_loss, message_loss, adv_loss, stego_images, recovered_messages, disc_pred
        torch.cuda.empty_cache()

        total_image_loss += image_loss.item()
        total_message_loss += message_loss.item()
        total_disc_loss += disc_loss.item()
        total_adv_loss += adv_loss.item()
        total_bit_accuracy += bit_accuracy.item()
        num_batches += 1
        global_step += 1

        # Promedio por epoch
        avg_image_loss = total_image_loss / num_batches
        avg_message_loss = total_message_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches
        avg_adv_loss = total_adv_loss / num_batches
        avg_bit_accuracy = total_bit_accuracy / num_batches

        if i % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}], Image Loss: {avg_image_loss:.4f}, Message Loss: {avg_message_loss:.4f}, "
                f"Disc Loss: {avg_disc_loss:.4f}, Adv Loss: {avg_adv_loss:.4f}, Bit Acc: {avg_bit_accuracy:.4f},")

    if (epoch + 1) % EPOCHS_TO_VAL == 0:
        evaluate_on_testset(encoder, decoder, discriminator, test_loader, writer, device, epoch)
        print("Evaluado sobre el test wey")

    # TensorBoard logging por epoch
    writer.add_scalar("Loss/Image", avg_image_loss, epoch)
    writer.add_scalar("Loss/Message", avg_message_loss, epoch)
    writer.add_scalar("Loss/Discriminator", avg_disc_loss, epoch)
    writer.add_scalar("Loss/Adversarial", avg_adv_loss, epoch)
    writer.add_scalar("Accuracy/Bit", avg_bit_accuracy, epoch)

    # Histogramas de pesos y gradientes
    for name, param in encoder.named_parameters():
        writer.add_histogram(f"Encoder/weights/{name}", param, global_step)
        if param.grad is not None:
            writer.add_histogram(f"Encoder/grads/{name}", param.grad, global_step)
    for name, param in decoder.named_parameters():
        writer.add_histogram(f"Decoder/weights/{name}", param, global_step)
        if param.grad is not None:
            writer.add_histogram(f"Decoder/grads/{name}", param.grad, global_step)
    for name, param in discriminator.named_parameters():
        writer.add_histogram(f"Discriminator/weights/{name}", param, global_step)
        if param.grad is not None:
            writer.add_histogram(f"Discriminator/grads/{name}", param.grad, global_step)

    # Visualización de imágenes reales vs stego
    img_grid_real = make_grid(images[:8].cpu(), nrow=4, normalize=True)
    img_grid_stego = make_grid(stego_images[:8].detach().cpu(), nrow=4, normalize=True)
    writer.add_image("Images/Real", img_grid_real, epoch)
    writer.add_image("Images/Stego", img_grid_stego, epoch)


    # Guardar modelos cada EPOCHS_TO_SAVE epochs
    if (epoch + 1) % EPOCHS_TO_SAVE == 0:
        torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, f"encoder_epoch{epoch+1}.pt"))
        torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, f"decoder_epoch{epoch+1}.pt"))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f"discriminator_epoch{epoch+1}.pt"))
        print(f"Modelos guardados en epoch {epoch+1}")

writer.close()