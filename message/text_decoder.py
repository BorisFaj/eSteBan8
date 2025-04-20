import random
import torch
import torch.nn as nn

class TextDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_len, dropout=0.3):
        super(TextDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, sos_token_id=None, targets=None, generate=False, teacher_forcing_ratio=0.5):
        batch_size = z.size(0)
        hidden = z.unsqueeze(0)  # (1, B, H)

        inputs = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=z.device)
        inputs = self.embedding(inputs)  # (B, 1, E)
        outputs = []

        seq_len = targets.size(1) if targets is not None else self.max_len
        for t in range(seq_len):
            out, hidden = self.gru(inputs, hidden)
            out_vocab = self.fc_out(out.squeeze(1))  # (B, Vocab)
            outputs.append(out_vocab.unsqueeze(1))

            if generate or targets is None or random.random() > teacher_forcing_ratio:
                top1 = out_vocab.argmax(1).unsqueeze(1)
            else:
                top1 = targets[:, t].unsqueeze(1)

            inputs = self.embedding(top1)

        return torch.cat(outputs, dim=1)  # (B, T, V)
