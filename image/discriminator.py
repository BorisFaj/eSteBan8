import torch
import torch.nn as nn
import torch.nn.init as init

class Discriminator(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.fc = None  # capa lineal definida en el primer forward

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming para activaciones tipo ReLU/LeakyReLU
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image):
        x = self.conv(image)
        B, C, H, W = x.shape
        x = x.view(B, -1)

        if self.fc is None:
            self.fc = nn.Linear(C * H * W, 1).to(x.device).half()
            init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
            init.constant_(self.fc.bias, 0)

        return torch.sigmoid(self.fc(x))
