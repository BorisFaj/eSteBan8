import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, image_channels, message_size, image_size):
        super().__init__()
        self.image_channels = image_channels
        self.message_size = message_size
        self.image_size = image_size

        self.conv1 = nn.Conv2d(image_channels + message_size, 64, 3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)  # Dropout aplicado por canal

        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(32, image_channels, 1)
        self.tanh = nn.Tanh()

    def forward(self, image, message):
        image = image.to(self.conv1.weight.device)
        message = message.to(self.conv1.weight.device)

        B, C, H, W = image.shape
        msg_map = message.view(B, self.message_size, 1, 1).expand(B, self.message_size, H, W)
        x = torch.cat([image, msg_map], dim=1)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.tanh(x)

        return x
