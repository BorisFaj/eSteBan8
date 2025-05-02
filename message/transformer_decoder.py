import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_len, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim, max_len)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
        self.max_len = max_len

    def forward(self, memory, sos_token_id, targets=None, generate=False, teacher_forcing_ratio=1.0, eos_token_id=None):
        B = memory.size(0)

        if generate:
            generated = torch.full((B, 1), sos_token_id, dtype=torch.long, device=memory.device)
            for _ in range(self.max_len - 1):
                tgt_embed = self.embedding(generated)
                tgt_embed = self.positional_encoding(tgt_embed)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1)).to(memory.device)

                output = self.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
                next_token_logits = self.output_proj(output[:, -1])
                next_token = next_token_logits.argmax(-1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)

                # Early stopping si todos generaron <eos>
                if eos_token_id is not None:
                    if (next_token == eos_token_id).all():
                        break

            return generated  # ⬅️ devuelve los IDs

        else:
            if targets is None:
                raise ValueError("Targets required when generate=False")
            tgt_embed = self.embedding(targets)
            tgt_embed = self.positional_encoding(tgt_embed)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(targets.size(1)).to(memory.device)

            output = self.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
            return self.output_proj(output)
