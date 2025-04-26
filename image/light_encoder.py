import torch
import torch.nn as nn
import torch.nn.init as init

class LightEncoder(nn.Module):
    def __init__(self, image_channels, message_size, image_size):
        super().__init__()
        self.image_channels = image_channels
        self.message_size = message_size
        self.image_size = image_size

        self.conv1 = nn.Conv2d(image_channels + message_size, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, image_channels, 3, padding=1)

        self.skip_conv = nn.Conv2d(image_channels + message_size, image_channels, 1)

        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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

        # Conexión residual
        skip = self.skip_conv(x_input)
        x = x + skip

        return self.tanh(x)
