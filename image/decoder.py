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
        self._fc_initialized = False  # Flag para inicialización diferida

    def forward(self, stego_image):
        x = self.conv(stego_image)
        B, C, H, W = x.shape
        x = x.reshape(B, -1)

        if not self._fc_initialized:
            self.fc = nn.Linear(C * H * W, self.message_size).to(x.device).half()
            self._fc_initialized = True
            self.add_module("fc", self.fc)  # Registra como parte del modelo
            print("🧠 Decoder creado")

        x = self.fc(x)
        return x
