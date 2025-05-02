import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(img_channels, 64, normalize=False),  # 112x112
            *block(64, 128),                           # 56x56
            *block(128, 256),                          # 28x28
            *block(256, 512),                          # 14x14
            nn.Conv2d(512, 1, 3, 1, 1),                # 14x14
            nn.AdaptiveAvgPool2d(1),                  # Global average
            nn.Flatten(),                             # (batch, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
