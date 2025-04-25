import torch.nn as nn
import torch.nn.init as init


class Decoder(nn.Module):
    def __init__(self, image_channels, message_size):
        super().__init__()
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.GroupNorm(4, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.GroupNorm(8, 64)

        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # Reducción razonable
        self.message_size = message_size
        self._fc_initialized = False

        for layer in [self.conv1, self.conv2]:
            init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, stego_image):
        x = self.relu1(self.conv1(stego_image))
        x = self.norm1(x)
        x = self.relu2(self.conv2(x))
        x = self.norm2(x)

        if stego_image.shape == x.shape:
            x = x + stego_image

        x = self.pool(x)  # B x 64 x 8 x 8
        B, C, H, W = x.shape
        x = x.view(B, -1)  # B x 4096

        if not self._fc_initialized:
            self.fc = nn.Linear(C * H * W, self.message_size).to(x.device)
            init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
            nn.init.constant_(self.fc.bias, 0)
            self._fc_initialized = True
            self.add_module("fc", self.fc)
            print("🧠 Decoder creado")

        return self.fc(x)
