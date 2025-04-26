import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from pytorch_msssim import ssim
from torch import amp
from light_encoder import LightEncoder
from light_decoder import LightDecoder
from style_loss import edge_loss
from data_handler import DataHandler
from dotenv import load_dotenv
import os
from start_experiment import start_mlflow, log_gpu_stats, log_model_histograms
import random


load_dotenv()

# Config
image_channels = int(os.getenv("image_channels"))
image_size = int(os.getenv("image_size"))
message_size = int(os.getenv("message_size"))
batch_size = int(os.getenv("batch_size"))
num_epochs = int(os.getenv("num_epochs"))
image_loss_lambda = float(os.getenv("image_loss_lambda"))
noise_std = float(os.getenv("noise_std"))
style_loss_weight = float(os.getenv("style_loss_weight"))
message_weight = float(os.getenv("message_weight"))
RUN_NAME = os.getenv("RUN_NAME") + "_autoencoder"
EPOCHS_TO_VAL = int(os.getenv("EPOCHS_TO_VAL"))  # numero de epochs entre validaciones
EPOCHS_TO_SAVE = int(os.getenv("EPOCHS_TO_SAVE"))  # numero de epochs para guardar el modelo
log_dir = os.path.join(os.getenv("log_dir"), f"{RUN_NAME}")
os.makedirs(log_dir, exist_ok=True)

# Cuda y precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
scaler = amp.GradScaler()

# Writer
writer = SummaryWriter(log_dir)

# Modelos
encoder = LightEncoder(image_channels=image_channels, message_size=message_size).to(dtype=torch.float32).to(device)
decoder = LightDecoder(image_channels=image_channels, message_size=message_size).to(dtype=torch.float32).to(device)
encoder = torch.compile(encoder)
decoder = torch.compile(decoder)

optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

# Datos
train_dataset, train_loader, val_dataset, val_loader = DataHandler(batch_size=batch_size).get()

# MLflow
mlflow = start_mlflow(params={
        "image_channels": image_channels,
        "image_size": image_size,
        "message_size": message_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "image_loss_lambda": image_loss_lambda,
        "noise_std": noise_std,
        "style_loss_weight": style_loss_weight,
        "message_weight": message_weight,
        "EPOCHS_TO_VAL": EPOCHS_TO_VAL,
        "EPOCHS_TO_SAVE": EPOCHS_TO_SAVE
    }, run_name=RUN_NAME)

for epoch in range(num_epochs):
    encoder.train()
    total_image_loss = 0
    total_message_loss = 0
    total_edge_loss = 0
    total_batches = 0

    for images, messages in train_loader:
        images = images.to(device)
        messages = messages.to(device)

        if noise_std > 0:
            noise = torch.randn_like(images) * noise_std
            images = torch.clamp(images + noise, -1, 1)
            assert torch.all(torch.isfinite(images)), "❌ Imagen con NaN o Inf antes del encoder"

        with amp.autocast("cuda", dtype=torch.float32):
            stego_images = encoder(images, messages)
            image_loss = (1 - ssim((stego_images + 1)/2, (images + 1)/2, data_range=1.0, size_average=True)) + \
                         image_loss_lambda * F.mse_loss(stego_images, images)

            decoded_msg = decoder(stego_images.detach())
            msg_loss = F.mse_loss(decoded_msg, messages)

            _edge_loss = edge_loss((stego_images + 1) / 2, (images + 1) / 2)
            total = image_loss + message_weight * msg_loss + style_loss_weight * _edge_loss


        if not torch.all(torch.isfinite(images)):
            print("🚨 imágenes corruptas")
        if not torch.all(torch.isfinite(stego_images)):
            print("🚨 stego corrupto")

        if torch.isnan(image_loss):
            print("⚠️ image_loss is NaN")
        if torch.isnan(_edge_loss):
            print("⚠️ style_loss is NaN")
        if torch.isnan(total):
            print("❌ NaN detected in total loss. Skipping batch.")
            print(f"image_loss: {image_loss.item()}, edge_loss: {_edge_loss.item()}")

            continue

        optimizer.zero_grad()
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        log_gpu_stats(mlflow=mlflow, epoch=epoch)  # lo pongo aqui que es cuando consume

        total_image_loss += image_loss.item()
        total_message_loss += msg_loss.item()
        total_edge_loss += _edge_loss.item()
        total_batches += 1

    torch.cuda.empty_cache()

    if total_batches == 0:
        print("⚠️ No batches processed due to NaNs. Stopping training.")
        break

    avg_image_loss = total_image_loss / total_batches
    avg_message_loss = total_message_loss / total_batches
    avg_style = total_edge_loss / total_batches

    print(f"Epoch {epoch + 1}/{num_epochs}, Image Loss: {avg_image_loss:.4f}, Style Loss: {avg_style:.4f}, Message Loss: {avg_message_loss:.4f}")
    writer.add_scalar("Loss/Image", avg_image_loss, epoch)
    writer.add_scalar("Loss/Style", avg_style, epoch)
    writer.add_scalar("Loss/Message", avg_message_loss, epoch)

    log_model_histograms(writer, encoder, "Encoder", epoch)
    log_model_histograms(writer, decoder, "Decoder", epoch)

    mlflow.log_metric("Loss/Image", avg_image_loss, step=epoch)
    mlflow.log_metric("Loss/Style", avg_style, step=epoch)
    mlflow.log_metric("Loss/Message", avg_message_loss, step=epoch)

    # Validacion cada EPOCHS_TO_VAL epochs
    if (epoch + 1) % EPOCHS_TO_VAL == 0:
        encoder.eval()
        with torch.no_grad():
            val_dataset = val_loader.dataset
            idx1, idx2 = random.sample(range(len(val_dataset)), 2)
            img1, _ = val_dataset[idx1]
            img2, _ = val_dataset[idx2]

            img = torch.stack([img1, img2], dim=0).to(device)
            msg = torch.randn(2, message_size).to(device)
            msg = msg / msg.norm(dim=1, keepdim=True)

            stego = encoder(img, msg)
            diff = (stego[0] - stego[1]).abs().mean().item()
            diff_map = ((stego[:1] - img[:1]) ** 2).mean(dim=1, keepdim=True)
            norm_diff = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

            writer.add_scalar("Debug/StegoMsgDiff", diff, epoch)
            writer.add_image("Debug/DiffMap", diff_map[0], epoch)
            writer.add_image("Debug/NormalizedDiffMap", norm_diff[0], epoch)
            mlflow.log_metric("Debug/StegoMsgDiff", diff, step=epoch)

        with torch.no_grad():
            val_img_loss = 0
            val_batches = 0
            val_msg_loss = 0
            for val_imgs, val_msgs in val_loader:
                val_imgs = val_imgs.to(device)
                val_msgs = val_msgs.to(device)
                with amp.autocast("cuda", dtype=torch.float32):
                    val_stego = encoder(val_imgs, val_msgs)

                    _img_loss = (1 - ssim((val_stego + 1) / 2, (val_imgs + 1) / 2, data_range=1.0, size_average=True)) + \
                                image_loss_lambda * F.mse_loss(val_stego, val_imgs)

                    val_decoded_msg = decoder(val_stego)

                    val_msg_loss += F.mse_loss(val_decoded_msg, val_msgs).item()
                    val_img_loss += _img_loss.item()
                    val_batches += 1

            avg_img_loss = val_img_loss / val_batches
            avg_msg_loss = val_msg_loss / val_batches
            print(f"\tValidation Image Loss: {avg_img_loss:.4f}")
            writer.add_scalar("Val/Loss/Image", avg_img_loss, epoch)
            writer.add_scalar("Val/Loss/Message", avg_msg_loss, epoch)
            mlflow.log_metric("Val/Loss/Image", avg_img_loss, step=epoch)
            mlflow.log_metric("Val/Loss/Message", avg_msg_loss, step=epoch)

            # Logs
            with amp.autocast("cuda", dtype=torch.float32):
                test_imgs, test_msgs = next(iter(val_loader))
                test_imgs = test_imgs.to(device)
                test_msgs = test_msgs.to(device).to(torch.float32)
                stego_imgs = encoder(test_imgs, test_msgs)
                img_real = make_grid((test_imgs[:8] + 1) / 2, nrow=4).cpu()
                img_stego = make_grid((stego_imgs[:8] + 1) / 2, nrow=4).cpu()
            writer.add_image("Autoencoder/Real", img_real, epoch)
            writer.add_image("Autoencoder/Stego", img_stego, epoch)

    if (epoch + 1) % EPOCHS_TO_SAVE == 0:
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),  # por si usas AMP
        }
        torch.save(checkpoint, f"{log_dir}/checkpoint_epoch_{epoch + 1}.pt")
        mlflow.log_artifact(f"{log_dir}/checkpoint_epoch_{epoch + 1}.pt")


writer.close()
mlflow.end_run()
print("✅ Entrenamiento del encoder como autoencoder finalizado.")