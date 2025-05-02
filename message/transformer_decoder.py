import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_len, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
        self.max_len = max_len

    def forward(self, memory, sos_token_id, targets=None, generate=False, teacher_forcing_ratio=1.0):
        B = memory.size(0)

        if generate:
            generated = torch.full((B, 1), sos_token_id, dtype=torch.long, device=memory.device)
            for _ in range(self.max_len - 1):
                tgt_embed = self.embedding(generated) + self.positional_encoding[:, :generated.size(1), :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1)).to(memory.device)

                output = self.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
                next_token_logits = self.output_proj(output[:, -1])
                next_token = next_token_logits.argmax(-1).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)

            return self.output_proj(self.embedding(generated) + self.positional_encoding[:, :generated.size(1), :])

        else:
            if targets is None:
                raise ValueError("Targets required when generate=False")
            tgt_inputs = targets  # usar directamente los targets (teacher forcing)
            pos_enc = self.positional_encoding[:, :tgt_inputs.size(1), :].clone().detach()
            tgt_embed = self.embedding(tgt_inputs) + pos_enc
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_inputs.size(1)).to(memory.device)

            output = self.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
            return self.output_proj(output)
