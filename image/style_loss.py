import torch
import torch.nn.functional as F
from torchvision.models import vgg19
from torch import nn
from torchvision.models import VGG19_Weights


class StyleLossHelper(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Cargamos VGG19 preentrenado
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].to(device)  # hasta conv3_3
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

        # Normalización según imagenNet
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(self, x):
        return (x + 1) / 2  # convertir de [-1,1] a [0,1]

    def normalize(self, x):
        return (x - self.mean) / self.std

    def gram_matrix(self, features):
        B, C, H, W = features.shape
        F_flat = features.view(B, C, H * W)
        G = torch.bmm(F_flat, F_flat.transpose(1, 2))  # BxCxC
        return G / (C * H * W)

    def forward(self, real, generated):
        real = self.normalize(self.preprocess(real))
        generated = self.normalize(self.preprocess(generated))

        feats_real = self.vgg(real)
        feats_gen = self.vgg(generated)

        gram_real = self.gram_matrix(feats_real)
        gram_gen = self.gram_matrix(feats_gen)

        loss = F.mse_loss(gram_gen, gram_real)
        return loss
