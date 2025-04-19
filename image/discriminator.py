import torch.nn as nn


# Discriminador: intenta distinguir entre imágenes reales y stego
class Discriminator(nn.Module):
    def __init__(self, image_channels):
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
