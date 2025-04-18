import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os

# Parámetros
image_channels = 3
image_size = 32
message_size = 512
batch_size = 64
num_epochs = 5000
log_dir = './runs/steganography_gan5'
checkpoint_dir = './checkpoints'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir)

# Encoder: imagen + mensaje -> stego_image
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels + message_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, image_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, image, message):
        msg_map = message.view(-1, message_size, 1, 1).expand(-1, message_size, image_size, image_size)

        x = torch.cat([image, msg_map], dim=1)
        return self.net(x)

# Decoder: stego_image -> mensaje
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * image_size * image_size, message_size),
            nn.Sigmoid()
        )

    def forward(self, stego_image):
        return self.net(stego_image)

# Discriminador: intenta distinguir entre imágenes reales y stego
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.net(image)

# Dataset CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Inicialización
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator = Discriminator().to(device)

enc_dec_opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

bce = nn.BCELoss()

# Entrenamiento
global_step = 0
for epoch in range(num_epochs):
    total_image_loss = 0
    total_message_loss = 0
    total_disc_loss = 0
    total_adv_loss = 0
    num_batches = 0

    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        messages = torch.randint(0, 2, (images.size(0), message_size)).float().to(device)

        # Paso forward
        stego_images = encoder(images, messages)
        recovered_messages = decoder(stego_images)

        # Discriminador
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        disc_real = discriminator(images)
        disc_fake = discriminator(stego_images.detach())

        disc_loss = bce(disc_real, real_labels) + bce(disc_fake, fake_labels)

        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()

        # Encoder + Decoder
        disc_pred = discriminator(stego_images)

        image_loss = 1 - ssim(stego_images, images, data_range=1.0, size_average=True)
        message_loss = bce(recovered_messages, messages)
        adv_loss = bce(disc_pred, real_labels)

        total_loss = image_loss + message_loss + adv_loss

        enc_dec_opt.zero_grad()
        total_loss.backward()
        enc_dec_opt.step()

        total_image_loss += image_loss.item()
        total_message_loss += message_loss.item()
        total_disc_loss += disc_loss.item()
        total_adv_loss += adv_loss.item()
        num_batches += 1
        global_step += 1

        # Promedio por época
        avg_image_loss = total_image_loss / num_batches
        avg_message_loss = total_message_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches
        avg_adv_loss = total_adv_loss / num_batches

        if i % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}], Image Loss: {image_loss.item():.4f}, Message Loss: {message_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")

    # TensorBoard logging por epoch
    writer.add_scalar("Loss/Image", avg_image_loss, epoch)
    writer.add_scalar("Loss/Message", avg_message_loss, epoch)
    writer.add_scalar("Loss/Discriminator", avg_disc_loss, epoch)
    writer.add_scalar("Loss/Adversarial", avg_adv_loss, epoch)

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




    # Guardar modelos cada 400 epochs
    if (epoch + 1) % 400 == 0:
        torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, f"encoder_epoch{epoch+1}.pt"))
        torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, f"decoder_epoch{epoch+1}.pt"))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f"discriminator_epoch{epoch+1}.pt"))
        print(f"Modelos guardados en epoch {epoch+1}")

writer.close()