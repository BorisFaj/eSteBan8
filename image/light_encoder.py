import torch
import torch.nn as nn
import torch.nn.init as init

class LightEncoder(nn.Module):
    def __init__(self, image_channels, message_size):
        super().__init__()
        self.conv = nn.Conv2d(image_channels + message_size, image_channels, 1)  # solo 1x1
        self.tanh = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image, message):
        image = image.to(self.conv.weight.device)
        message = message.to(self.conv.weight.device)

        B, C, H, W = image.shape
        msg_map = message.view(B, -1, 1, 1).expand(B, message.shape[1], H, W)
        x_input = torch.cat([image, msg_map], dim=1)

        x = self.conv(x_input)
        return self.tanh(x)