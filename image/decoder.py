import torch.nn as nn


class Decoder(nn.Module):
# Decoder: stego_image -> mensaje
    def __init__(self, image_channels, image_size, message_size):
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
