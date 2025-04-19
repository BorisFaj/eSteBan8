import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, image_channels, message_size, image_size):
        super().__init__()
        self.image_channels = image_channels
        self.message_size = message_size
        self.image_size = image_size

        self.net = nn.Sequential(
            nn.Conv2d(image_channels + message_size, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, image_channels, 1),
            nn.Tanh()
        )

    def forward(self, image, message):
        B, C, H, W = image.shape
        msg_map = message.view(-1, self.message_size, 1, 1).expand(-1, self.message_size, H, W)
        x = torch.cat([image, msg_map], dim=1)

        return self.net(x)