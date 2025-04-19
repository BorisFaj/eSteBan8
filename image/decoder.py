import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, image_channels, message_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.message_size = message_size
        self.fc = None  # Se definirá dinámicamente en el primer forward

    def forward(self, stego_image):
        x = self.conv(stego_image)
        B, C, H, W = x.shape
        x = x.view(B, -1)

        # Inicializar la capa fc si aún no está definida
        if self.fc is None:
            self.fc = nn.Linear(C * H * W, self.message_size).to(x.device)

        x = self.fc(x)
        return x
