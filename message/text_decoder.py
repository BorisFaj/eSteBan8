import torch
import torch.nn as nn


class TextDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_len=30):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

        self.latent_to_hidden = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z, sos_token_id=101, generate=False):
        """
        :param z: tensor (batch_size, hidden_dim)
        :param generate: si True, devuelve los IDs generados; si False, devuelve logits para entrenamiento
        :return: (batch_size, max_len, vocab_size) si generate=False
                 (batch_size, max_len) si generate=True
        """
        batch_size = z.size(0)
        if z.dim() == 2:
            hidden = self.latent_to_hidden(z).unsqueeze(0)
        elif z.dim() == 3:
            hidden = self.latent_to_hidden(z).squeeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Expected z with 2 or 3 dims, got shape: {z.shape}")

        inputs = torch.full((batch_size, 1), sos_token_id, dtype=torch.long).to(z.device)

        logits_list = []
        token_ids = []

        for _ in range(self.max_len):
            embedded = self.embedding(inputs)              # (B, 1, emb_dim)
            out, hidden = self.gru(embedded, hidden)       # out: (B, 1, hidden_dim)
            logits = self.out(out[:, -1, :])               # (B, vocab_size)
            logits_list.append(logits.unsqueeze(1))        # (B, 1, vocab_size)

            if generate:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
                token_ids.append(next_token)
                inputs = next_token
            else:
                inputs = torch.argmax(logits, dim=-1, keepdim=True)  # teacher forcing opcional

        if generate:
            return torch.cat(token_ids, dim=1)             # (B, max_len)
        else:
            return torch.cat(logits_list, dim=1)           # (B, max_len, vocab_size)
