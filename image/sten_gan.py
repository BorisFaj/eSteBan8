import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F
from light_encoder import LightEncoder
from light_decoder import LightDecoder
from discriminator import Discriminator
from torch import amp
import math
from dotenv import load_dotenv
import os
from data_handler import DataHandler
from start_experiment import start_mlflow, log_gpu_stats, log_model_histograms
load_dotenv()

# Cuda
torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Scaler
scaler = amp.GradScaler()

# Parametros de las redes
WARM_UP_LEN = int(os.getenv("WARM_UP_LEN"))  # numero de epochs que dejo al discriminador sin entrenar
image_loss_lambda = float(os.getenv("image_loss_lambda"))  # Parametro para darle algo de tolerancia al image loss
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
style_loss_weight = float(os.getenv("style_loss_weight"))
message_weight = float(os.getenv("message_weight"))
disc_loss_target = float(os.getenv("disc_loss_target"))
message_loss_target = float(os.getenv("message_loss_target"))
sharpness = float(os.getenv("sharpness"))

RUN_NAME = os.getenv("RUN_NAME")

# Checkpoints
checkpoint_dir = os.getenv("checkpoint_dir")
os.makedirs(checkpoint_dir, exist_ok=True)

# Tensorboard
log_dir = os.getenv("log_dir")
log_dir = os.path.join(log_dir, RUN_NAME)
os.makedirs(log_dir, exist_ok=True)

# MLFlow
mlflow = start_mlflow({"warm_up_len": WARM_UP_LEN,
                       "batch_size": batch_size,
                       "message_size": message_size,
                       "image_loss_lambda": image_loss_lambda,
                       "freeze_disc_loss": FREEZE_DISC_LOSS,
                       "image_channels": image_channels,
                       "image_size": image_size,
                       "num_epochs": num_epochs,
                       "image_input_res": IMAGE_INPUT_RES,
                       "noise_std": noise_std,
                       "style_loss_weight": style_loss_weight,
                       "message_weight": message_weight,
                       "EPOCHS_TO_VAL": EPOCHS_TO_VAL,
                       "EPOCHS_TO_SAVE": EPOCHS_TO_SAVE,
                       "disc_loss_target": disc_loss_target,
                       "message_loss_target": message_loss_target,
                       "sharpness": sharpness
                       },
                      run_name=RUN_NAME)

def should_train_discriminator(message_loss: float, disc_loss: float, writer, epoch, message_loss_target: float,
                               disc_loss_target: float, sharpness: float) -> bool:
    """
    Calcula una probabilidad suave de entrenar el discriminador.

    - Si sharpness es alto (e.g. 10), el cambio es brusco (como una puerta).
    - Si sharpness es bajo (e.g. 2), el cambio es progresivo (margen amplio).

    Devuelve un número entre 0 y 1.
    """

    # Normalizamos respecto a los objetivos
    message_factor = max(0.0, 1.0 - (message_loss / message_loss_target))
    disc_factor = max(0.0, 1.0 - (disc_loss / disc_loss_target))

    # Combinamos (puedes ajustar pesos si quieres)
    score = 0.5 * message_factor + 0.5 * disc_factor

    # Aplicamos función sigmoide controlada por sharpness
    probability = 1 / (1 + math.exp(-sharpness * (score - 0.5)))

    # Asegurar que está en [0, 1]
    probability = min(max(probability, 0.0), 1.0)

    writer.add_scalar("Debug/Discriminator_Train_Prob", probability, epoch)

    return torch.rand(1).item() < probability


def evaluate_on_testset(encoder, decoder, discriminator, test_loader, writer, device, epoch):
    encoder.eval()
    decoder.eval()
    discriminator.eval()

    total_message_loss = 0
    total_adv_loss = 0
    total_bit_accuracy = 0
    total_disc_loss = 0
    num_batches = 0

    bce = nn.BCEWithLogitsLoss()

    with torch.no_grad(), amp.autocast("cuda"):
        for i, (images, messages) in enumerate(test_loader):
            images = images.to(device)
            messages = messages.to(device)

            # Forward
            stego_images = encoder(images, messages)
            recovered_messages = decoder(stego_images)
            disc_pred = discriminator(stego_images)

            # Métricas
            disc_real = discriminator(images)
            disc_fake = discriminator(stego_images)

            disc_loss = F.mse_loss(disc_real, torch.ones_like(disc_real)) + \
                        F.mse_loss(disc_fake, torch.zeros_like(disc_fake))

            message_loss = bce(recovered_messages, messages)

            pred_bits = (torch.sigmoid(recovered_messages) > 0.5).int()
            true_bits = messages.int().to(pred_bits.device)
            bit_accuracy = (pred_bits == true_bits).float().mean()

            adv_loss = F.mse_loss(disc_pred, torch.ones_like(disc_pred))

            total_message_loss += message_loss.item()
            total_adv_loss += adv_loss.detach().item()
            total_bit_accuracy += bit_accuracy.item()
            total_disc_loss += disc_loss.item()


            num_batches += 1

        avg_message_loss = total_message_loss / num_batches
        avg_bit_accuracy = total_bit_accuracy / num_batches
        avg_adv_loss = total_adv_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches

        # MLFlow logging por epoch
        mlflow.log_metric("Test/Loss/Message", avg_message_loss, step=epoch)
        mlflow.log_metric("Test/Loss/Discriminator", avg_disc_loss, step=epoch)
        mlflow.log_metric("Test/Accuracy/Bit", avg_bit_accuracy, step=epoch)
        mlflow.log_metric("Test/Accuracy/Adversarial", avg_adv_loss, step=epoch)

        # TensorBoard logging por epoch
        writer.add_scalar("Test/Loss/Message", avg_message_loss, epoch)
        writer.add_scalar("Test/Loss/Discriminator", avg_disc_loss, epoch)
        writer.add_scalar("Test/Loss/Adversarial", avg_adv_loss, epoch)
        writer.add_scalar("Test/Accuracy/Bit", avg_bit_accuracy, epoch)


        # Imágenes
        images_01 = (images + 1) / 2
        stego_images_01 = (stego_images + 1) / 2
        img_grid_real = make_grid(images_01[:8].cpu(), nrow=4, normalize=True)
        img_grid_stego = make_grid(stego_images_01[:8].cpu(), nrow=4, normalize=True)
        writer.add_image("Test/Images/Real", img_grid_real, epoch)
        writer.add_image("Test/Images/Stego", img_grid_stego, epoch)

        # Debug de diferencias entre stego-images
        img = images[:2]  # coge dos imágenes del batch
        stego = stego_images[:2]

        diff = (stego[0] - stego[1]).abs().mean().item()
        diff_map = ((stego[:1] - img[:1]) ** 2).mean(dim=1, keepdim=True)
        norm_diff = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

        writer.add_image("Debug/Real", img[0].cpu(), epoch)
        writer.add_image("Debug/Stego", stego[0].cpu(), epoch)
        writer.add_image("Debug/Stego_vs_Real_DiffMap", diff_map[0], epoch)
        writer.add_image("Debug/NormalizedDiffMap", norm_diff[0], epoch)
        writer.add_scalar("Debug/Stego1_vs_Stego2_MsgDiff", diff, epoch)
        mlflow.log_metric("Debug/Stego1_vs_Stego2_MsgDiff", diff, step=epoch)

    encoder.train()
    decoder.train()
    discriminator.train()

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
writer.add_text("Entrenamiento", "Iniciado correctamente", 0)
writer.flush()

train_dataset, train_loader, test_dataset, test_loader = DataHandler(batch_size=batch_size).get()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = LightEncoder(image_channels=image_channels, message_size=message_size).to(device)
decoder = LightDecoder(image_channels=image_channels, message_size=message_size).to(device)
discriminator = Discriminator(image_channels=image_channels).to(device)

enc_dec_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

bce = nn.BCEWithLogitsLoss()

global_step = 0
# start_epoch = load_latest_checkpoint(checkpoint_dir, RUN_NAME, encoder, decoder, discriminator)
start_epoch = 0

# Empieza la marcha
for epoch in range(start_epoch, num_epochs):
    total_image_loss = 0
    total_message_loss = 0
    total_disc_loss = 0
    total_adv_loss = 0
    num_batches = 0
    disc_batches = 0
    total_bit_accuracy = 0
    train_discriminator = False

    for i, (images, messages) in enumerate(train_loader):
        images = get_noisy(images).to(device)
        messages = messages.to(device)

        if noise_std > 0:
            noise = torch.randn_like(images) * noise_std
            images = images + noise
            images = torch.clamp(images, -1, 1)  # mantén en el rango [-1, 1]

        adv_loss = F.mse_loss(torch.ones_like(torch.tensor([0.])), torch.ones_like(torch.tensor([0.])))  # 0

        # Forward
        with amp.autocast("cuda"):
            stego_images = encoder(images, messages)
            recovered_messages = decoder(stego_images)

            message_loss = bce(recovered_messages, messages)

            disc_real = discriminator(images)
            disc_fake = discriminator(stego_images.detach())

            disc_loss = F.mse_loss(disc_real, torch.ones_like(disc_real)) + \
                        F.mse_loss(disc_fake, torch.zeros_like(disc_fake))

            if train_discriminator:
                disc_opt.zero_grad()
                scaler.scale(disc_loss).backward()
                scaler.step(disc_opt)
                scaler.update()

                total_disc_loss += disc_loss.item()
                disc_batches += 1

                if not should_train_discriminator(message_loss= message_loss.item(),
                                                  disc_loss=disc_loss.item(),
                                                  writer=writer,
                                                  epoch=epoch,
                                                  disc_loss_target=disc_loss_target,
                                                  message_loss_target=message_loss_target,
                                                  sharpness=sharpness):
                    print("🧠 [Discriminador]: Paro de entrenar")
                    train_discriminator = False  # Deja de entrenar

            else: # si no esta entrenando
                if epoch > WARM_UP_LEN and should_train_discriminator(message_loss= message_loss.item(),
                                                  disc_loss=disc_loss.item(),
                                                  writer=writer,
                                                  epoch=epoch,
                                                  disc_loss_target=disc_loss_target,
                                                  message_loss_target=message_loss_target,
                                                  sharpness=sharpness):
                    print("🧠 [Discriminador]: empiezo a entrenar")
                    train_discriminator = True  # Empieza a entrenar

            # Resto de perdidas
            disc_pred = discriminator(stego_images)
            adv_loss = F.mse_loss(disc_pred, torch.ones_like(disc_pred))
            total_loss = message_weight * message_loss + adv_loss

            enc_dec_opt.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(enc_dec_opt)
            scaler.update()
            log_gpu_stats(mlflow=mlflow, epoch=epoch)

        with torch.no_grad():
            # Bit Accuracy
            pred_bits = (torch.sigmoid(recovered_messages) > 0.5).int()
            true_bits = messages.int().to(pred_bits.device)
            bit_accuracy = (pred_bits == true_bits).float().mean()

            # Image loss
            images_32 = images.float()
            stego_32 = stego_images.float()
            qq = (1 - ssim(stego_32, images_32, data_range=1.0, size_average=True)) + \
                         image_loss_lambda * F.mse_loss(stego_images, images)


        total_message_loss += message_loss.item()

        total_adv_loss += adv_loss.item()
        total_bit_accuracy += bit_accuracy.item()
        num_batches += 1
        global_step += 1

        del total_loss, message_loss, adv_loss, recovered_messages, disc_pred
        torch.cuda.empty_cache()

    # Promedio por epoch
    if disc_batches > 0:
        avg_disc_loss = total_disc_loss / disc_batches
    else:
        avg_disc_loss = 0
    avg_message_loss = total_message_loss / num_batches
    avg_adv_loss = total_adv_loss / num_batches
    avg_bit_accuracy = total_bit_accuracy / num_batches


    if (epoch + 1) % EPOCHS_TO_VAL == 0:
        evaluate_on_testset(encoder, decoder, discriminator, test_loader, writer, device, epoch)
        print("Evaluado sobre el test wey")

    # MLFlow logging por epoch
    mlflow.log_metric("Loss/Message", avg_message_loss, step=epoch)
    mlflow.log_metric("Loss/Discriminator", avg_disc_loss, step=epoch)
    mlflow.log_metric("Accuracy/Bit", avg_bit_accuracy, step=epoch)
    mlflow.log_metric("Accuracy/Adversarial", avg_adv_loss, step=epoch)

    # TensorBoard logging por epoch
    writer.add_scalar("Loss/Message", avg_message_loss, epoch)
    writer.add_scalar("Loss/Discriminator", avg_disc_loss, epoch)
    writer.add_scalar("Loss/Adversarial", avg_adv_loss, epoch)
    writer.add_scalar("Accuracy/Bit", avg_bit_accuracy, epoch)
    log_model_histograms(writer, encoder, "Encoder", epoch)
    log_model_histograms(writer, decoder, "Decoder", epoch)
    log_model_histograms(writer, discriminator, "Discriminator", epoch)

    images_01 = (images + 1) / 2
    stego_images_01 = (stego_images + 1) / 2
    img_grid_real = make_grid(images_01[:8].cpu(), nrow=4, normalize=False)
    img_grid_stego = make_grid(stego_images_01[:8].detach().cpu(), nrow=4, normalize=False)

    writer.add_image("Images/Real", img_grid_real, epoch)
    writer.add_image("Images/Stego", img_grid_stego, epoch)

    # Guardar modelos cada EPOCHS_TO_SAVE epochs
    if (epoch + 1) % EPOCHS_TO_SAVE == 0:
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "scaler_state_dict": scaler.state_dict(),  # por si usas AMP
        }
        torch.save(checkpoint, f"{log_dir}/checkpoint_epoch_{epoch + 1}.pt")
        mlflow.log_artifact(f"{log_dir}/checkpoint_epoch_{epoch + 1}.pt")

        print(f"Modelos guardados en MLFlow")

mlflow.end_run()
writer.close()
