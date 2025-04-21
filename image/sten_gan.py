import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F
import os
from decoder import Decoder
from encoder import Encoder
from discriminator import Discriminator
from torch import amp
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

scaler = amp.GradScaler()

# Parámetros
WARM_UP_LEN = 50  # numero de epochs que dejo al discriminador sin entrenar
image_loss_lambda = 0.9  # Parametro para darle algo de tolerancia al image loss
DISC_FREEZE_WINDOW = 15  # ventana MAXIMA de epochs que se queda sin entrenar el discriminador despues del warmup
FREEZE_DISC_LOSS = 0.3  # loss maximo que alcanza el discriminador antes de ser congelado


def freeze_disc(global_step: int, epoch: int, k: float=0.005) -> int:

    freeze_window = max(1, int(DISC_FREEZE_WINDOW * math.exp(-k * epoch)))
    return global_step % freeze_window == 0

image_channels = 3
image_size = 32
message_size = 128  # Aumentar a 512
batch_size = 2
num_epochs = 5000
IMAGE_INPUT_RES = 128  # resolucion de la imagen de entrada
EPOCHS_TO_VAL = 20  # numero de epochs entre validaciones
EPOCHS_TO_SAVE = 10  # numero de epochs para guardar el modelo
RUN_NAME = "eSteBan8"
log_dir = f'./runs/{RUN_NAME}'
checkpoint_dir = f'./checkpoints/{RUN_NAME}'

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir)


# Dataset Open Images (resolución (IMAGE_INPUT_RES, IMAGE_INPUT_RES))
transform = transforms.Compose([
    transforms.Resize((IMAGE_INPUT_RES, IMAGE_INPUT_RES)),
    transforms.CenterCrop((IMAGE_INPUT_RES, IMAGE_INPUT_RES)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root="openimages_custom/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root="openimages_custom/val", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Inicialización
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(image_channels=image_channels, message_size=message_size, image_size=image_size).to(device)
decoder = Decoder(image_channels=image_channels, message_size=message_size).to(device)
discriminator = Discriminator(image_channels=image_channels).to(device)

enc_dec_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

bce = nn.BCEWithLogitsLoss()

# TEST
def evaluate_on_testset(encoder, decoder, discriminator, test_loader, writer, device, epoch):
    encoder.eval()
    decoder.eval()
    discriminator.eval()

    total_message_loss = 0
    total_image_loss = 0
    total_bit_accuracy = 0
    num_batches = 0

    bce = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            messages = torch.randint(0, 2, (images.size(0), message_size)).float().to(device)

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
            true_bits = messages.int()
            bit_accuracy = (pred_bits == true_bits).float().mean()

            total_message_loss += message_loss.item()
            total_image_loss += image_loss.detach().item()
            total_bit_accuracy += bit_accuracy.item()
            num_batches += 1

        avg_message_loss = total_message_loss / num_batches
        avg_image_loss = total_image_loss / num_batches
        avg_bit_accuracy = total_bit_accuracy / num_batches

        writer.add_scalar("Test/Loss/Message", avg_message_loss, epoch)
        writer.add_scalar("Test/Loss/Image", avg_image_loss, epoch)
        writer.add_scalar("Test/Accuracy/Bit", avg_bit_accuracy, epoch)

        # Imágenes ejemplo
        img_grid_real = make_grid(images_01[:8].cpu(), nrow=4, normalize=True)
        img_grid_stego = make_grid(stego_images_01[:8].cpu(), nrow=4, normalize=True)
        writer.add_image("Test/Images/Real", img_grid_real, epoch)
        writer.add_image("Test/Images/Stego", img_grid_stego, epoch)

    encoder.train()
    decoder.train()
    discriminator.train()


def load_latest_checkpoint(checkpoint_dir, encoder, decoder, discriminator, enc_dec_opt, disc_opt):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") and "encoder" in f]
    if not checkpoints:
        return 0  # No checkpoint found, start from epoch 0

    # Extraer el número de epoch del nombre de archivo
    get_epoch = lambda f: int(f.split("_epoch")[1].split(".pt")[0])
    latest_epoch = max(get_epoch(f) for f in checkpoints)

    print(f"🔁 Cargando checkpoint del epoch {latest_epoch}")

    encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"encoder_epoch{latest_epoch}.pt")))
    decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"decoder_epoch{latest_epoch}.pt")), strict=False)
    discriminator.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"discriminator_epoch{latest_epoch}.pt")))

    return latest_epoch


# Entrenamiento
global_step = 0
start_epoch = load_latest_checkpoint(checkpoint_dir, encoder, decoder, discriminator, enc_dec_opt, disc_opt)
for epoch in range(start_epoch, num_epochs):
    total_image_loss = 0
    total_message_loss = 0
    total_disc_loss = 0
    total_adv_loss = 0
    num_batches = 0
    total_bit_accuracy = 0

    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        messages = torch.randint(0, 2, (images.size(0), message_size)).float().to(device)

        # Paso forward
        torch.cuda.empty_cache()
        with amp.autocast("cuda"):
            stego_images = encoder(images, messages)
            recovered_messages = decoder(stego_images)

            # Bit Accuracy
            with torch.no_grad():
                pred_bits = (torch.sigmoid(recovered_messages) > 0.5).int()
                true_bits = messages.int()
                bit_accuracy = (pred_bits == true_bits).float().mean()

            # Discriminador
            real_labels = torch.full((images.size(0), 1), 0.9, device=device)
            fake_labels = torch.full((images.size(0), 1), 0.1, device=device)

            disc_real = discriminator(images)
            disc_fake = discriminator(stego_images.detach())

            disc_loss = F.mse_loss(disc_real, torch.ones_like(disc_real)) + \
                        F.mse_loss(disc_fake, torch.zeros_like(disc_fake))

            if disc_loss.item() <= FREEZE_DISC_LOSS:
                train_discriminator = False
            else:
                train_discriminator = True

            # El discriminador NO se entrena todos los epochs
            if train_discriminator:
                if epoch > WARM_UP_LEN and freeze_disc(global_step, epoch):
                    disc_opt.zero_grad()
                    with amp.autocast("cuda"):
                        disc_loss = F.mse_loss(disc_real, torch.ones_like(disc_real)) + \
                                    F.mse_loss(disc_fake, torch.zeros_like(disc_fake))
                    scaler.scale(disc_loss).backward()
                    scaler.step(disc_opt)
                    scaler.update()

            # Encoder + Decoder
            disc_pred = discriminator(stego_images)
            image_loss = (1 - ssim(stego_images, images, data_range=1.0, size_average=True)) + \
                         image_loss_lambda * F.mse_loss(stego_images, images)

            message_loss = bce(recovered_messages, messages)
            adv_loss = F.mse_loss(disc_pred, torch.ones_like(disc_pred))


            # Si no estamos entrenando el discriminador, no lo metemos en el total_loss
            if train_discriminator:
                _adv_loss = adv_loss
            else:
                _adv_loss = 0

            # WarmUP
            if epoch < WARM_UP_LEN:
                total_loss = message_loss
            else:
                total_loss = message_loss + _adv_loss

        enc_dec_opt.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(enc_dec_opt)
        scaler.update()

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