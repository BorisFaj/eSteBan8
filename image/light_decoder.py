import torch
import torch.nn as nn
import torch.nn.init as init


class LightDecoder(nn.Module):
    def __init__(self, image_channels, message_size):
        super().__init__()
        self.conv = nn.Conv2d(image_channels, message_size, 1)  # Inverso del encoder
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Comprime HxW a 1x1
        self.flatten = nn.Flatten()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x
