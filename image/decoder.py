import torch
import torch.nn as nn
import torch.nn.init as init

class Decoder(nn.Module):
    def __init__(self, image_channels, message_size):
        super().__init__()
        self.message_size = message_size

        self.conv1 = nn.Conv2d(image_channels, 64, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 64)
        self.act1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)
        self.act2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.norm3 = nn.GroupNorm(4, 32)
        self.act3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm4 = nn.GroupNorm(4, 32)
        self.act4 = nn.LeakyReLU(0.2)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Reducimos pero no matamos el espacio entero
        self.fc = nn.Linear(32 * 4 * 4, message_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Primer bloque
        skip = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        # Residual connection (conv1+conv2) + skip directo
        if skip.shape == x.shape:
            x = x + skip

        # Segundo bloque
        skip2 = x
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act4(x)

        if skip2.shape == x.shape:
            x = x + skip2

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
