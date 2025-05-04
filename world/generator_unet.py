import torch
import torch.nn as nn
import torch.nn.init as init

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_norm=True):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        else:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class GeneratorUNetCompact(nn.Module):
    def __init__(self, image_channels=3, base_channels=64):
        super().__init__()
        self.encoder1 = ConvBlock(image_channels, base_channels, down=True, use_norm=False)  # 64x64
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)  # 32x32
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)  # 16x16
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8)  # 8x8

        self.decoder1 = ConvBlock(base_channels * 8, base_channels * 4, down=False)
        self.decoder2 = ConvBlock(base_channels * 8, base_channels * 2, down=False)  # skip from encoder3
        self.decoder3 = ConvBlock(base_channels * 4, base_channels, down=False)  # skip from encoder2
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.encoder1(x)  # [B, 64, 64, 64]
        e2 = self.encoder2(e1)  # [B, 128, 32, 32]
        e3 = self.encoder3(e2)  # [B, 256, 16, 16]
        e4 = self.encoder4(e3)  # [B, 512, 8, 8]

        d1 = self.decoder1(e4)  # [B, 256, 16, 16]
        d2 = self.decoder2(torch.cat([d1, e3], dim=1))  # [B, 128, 32, 32]
        d3 = self.decoder3(torch.cat([d2, e2], dim=1))  # [B, 64, 64, 64]
        out = self.decoder4(torch.cat([d3, e1], dim=1))  # [B, 3, 128, 128]

        return out
