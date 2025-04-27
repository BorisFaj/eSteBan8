import torch
import torch.nn as nn
import torch.nn.init as init


class Encoder(nn.Module):
    def __init__(self, image_channels, message_size):
        super().__init__()
        self.image_channels = image_channels
        self.message_size = message_size

        # Capas convolucionales
        self.conv1 = nn.Conv2d(image_channels + message_size, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, image_channels, 1)

        # Conexión residual para ajustar canales
        self.skip_conv = nn.Conv2d(image_channels + message_size, image_channels, 1)

        # Activaciones y regularización
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(p=0.2)
        self.tanh = nn.Tanh()

        # Inicialización de pesos
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming para activaciones tipo ReLU/LeakyReLU
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, message):
        image = image.to(self.conv1.weight.device)
        message = message.to(self.conv1.weight.device)

        B, C, H, W = image.shape
        msg_map = message.view(B, self.message_size, 1, 1).expand(B, self.message_size, H, W)
        x_input = torch.cat([image, msg_map], dim=1)

        x = self.conv1(x_input)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)

        # Conexión residual
        skip = self.skip_conv(x_input)
        x = x + skip

        return self.tanh(x)
