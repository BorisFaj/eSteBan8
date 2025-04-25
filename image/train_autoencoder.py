import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from pytorch_msssim import ssim
from torch import amp
from encoder import Encoder
from style_loss import StyleLossHelper
from data_handler import DataHandler
from dotenv import load_dotenv
import os
from start_experiment import start_mlflow

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
RUN_NAME = os.getenv("RUN_NAME")
log_dir = os.path.join(os.getenv("log_dir"), f"{RUN_NAME}_autoencoder")
os.makedirs(log_dir, exist_ok=True)

# Cuda y precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
scaler = amp.GradScaler()

# Writer
writer = SummaryWriter(log_dir)

# Modelos
encoder = Encoder(image_channels=image_channels, message_size=message_size, image_size=image_size).to(device)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
style_loss_helper = StyleLossHelper(device)

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
        "style_loss_weight": style_loss_weight
    } , run_name="SteGAuto")


for epoch in range(num_epochs):
    encoder.train()
    total_loss = 0
    total_style_loss = 0
    total_batches = 0

    for images, messages in train_loader:
        images = images.to(device)
        messages = messages.to(device)

        if noise_std > 0:
            noise = torch.randn_like(images) * noise_std
            images = torch.clamp(images + noise, -1, 1)

        with amp.autocast("cuda"):
            stego_images = encoder(images, messages)
            image_loss = (1 - ssim((stego_images + 1)/2, (images + 1)/2, data_range=1.0, size_average=True)) + \
                         image_loss_lambda * F.mse_loss(stego_images, images)
            style_loss = style_loss_helper((images + 1)/2, (stego_images + 1)/2)
            total = image_loss + style_loss_weight * style_loss

        optimizer.zero_grad()
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += image_loss.item()
        total_style_loss += style_loss.item()
        total_batches += 1

    avg_loss = total_loss / total_batches
    avg_style = total_style_loss / total_batches

    print(f"Epoch {epoch + 1}/{num_epochs}, Image Loss: {avg_loss:.4f}, Style Loss: {avg_style:.4f}")
    writer.add_scalar("Loss/Image", avg_loss, epoch)
    writer.add_scalar("Loss/Style", avg_style, epoch)
    mlflow.log_metric("train_image_loss", avg_loss, step=epoch)
    mlflow.log_metric("train_style_loss", avg_style, step=epoch)

    # Validación cada 4 épocas
    if epoch % 4 == 0:
        encoder.eval()
        with torch.no_grad():
            val_loss = 0
            val_batches = 0
            for val_imgs, val_msgs in val_loader:
                val_imgs = val_imgs.to(device)
                val_msgs = val_msgs.to(device)
                val_stego = encoder(val_imgs, val_msgs)

                v_loss = (1 - ssim((val_stego + 1)/2, (val_imgs + 1)/2, data_range=1.0, size_average=True)) + \
                         image_loss_lambda * F.mse_loss(val_stego, val_imgs)
                val_loss += v_loss.item()
                val_batches += 1

            avg_val_loss = val_loss / val_batches
            print(f"\tValidation Image Loss: {avg_val_loss:.4f}")
            writer.add_scalar("Loss/Val_Image", avg_val_loss, epoch)
            mlflow.log_metric("val_image_loss", avg_val_loss, step=epoch)

    if epoch % 5 == 0:
        encoder.eval()
        with torch.no_grad():
            test_imgs, test_msgs = next(iter(val_loader))
            test_imgs = test_imgs.to(device)
            test_msgs = test_msgs.to(device)
            stego_imgs = encoder(test_imgs, test_msgs)
            img_real = make_grid((test_imgs[:8] + 1) / 2, nrow=4).cpu()
            img_stego = make_grid((stego_imgs[:8] + 1) / 2, nrow=4).cpu()
            writer.add_image("Autoencoder/Real", img_real, epoch)
            writer.add_image("Autoencoder/Stego", img_stego, epoch)

mlflow.pytorch.log_model(encoder, "encoder_model")

writer.close()
mlflow.end_run()
print("✅ Entrenamiento del encoder como autoencoder finalizado.")
