import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextCompressorVAE(nn.Module):
    def __init__(self, latent_dim=512, pooling='cls', freeze_bert=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.pooling = pooling
        self.fc_mu = nn.Linear(768, latent_dim)
        self.fc_logvar = nn.Linear(768, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        outputs = self.bert(**inputs)
        hidden_states = outputs.last_hidden_state

        if self.pooling == 'mean':
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states[:, 0, :]

        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar
