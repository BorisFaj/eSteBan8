import torch
import torch.nn as nn

class TransformerTextDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_len, n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, memory, sos_token_id, targets=None, generate=False, teacher_forcing_ratio=0.5):
        batch_size = memory.size(0)
        device = memory.device

        inputs = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
        outputs = []

        if generate:
            for _ in range(self.max_len):
                embedded = self.embedding(inputs)
                embedded = self.pos_encoder(embedded)

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(embedded.size(1)).to(device)

                out = self.transformer_decoder(tgt=embedded, memory=memory.unsqueeze(1), tgt_mask=tgt_mask)
                logits = self.fc_out(out[:, -1, :])  # solo el último paso
                outputs.append(logits.unsqueeze(1))

                next_token = logits.argmax(dim=-1).unsqueeze(1)
                inputs = torch.cat([inputs, next_token], dim=1)

            outputs = torch.cat(outputs, dim=1)
            return outputs

        else:
            seq_len = targets.size(1)
            for t in range(seq_len):
                embedded = self.embedding(inputs)
                embedded = self.pos_encoder(embedded)

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(embedded.size(1)).to(device)

                out = self.transformer_decoder(tgt=embedded, memory=memory.unsqueeze(1), tgt_mask=tgt_mask)
                logits = self.fc_out(out[:, -1, :])
                outputs.append(logits.unsqueeze(1))

                if random.random() < teacher_forcing_ratio:
                    next_input = targets[:, t].unsqueeze(1)
                else:
                    next_input = logits.argmax(dim=-1).unsqueeze(1)

                inputs = torch.cat([inputs, next_input], dim=1)

            outputs = torch.cat(outputs, dim=1)
            return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
