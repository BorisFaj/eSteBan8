import torch
import torch.nn as nn
import random


class LSTMAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_len, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)  # porque concat h_t + contexto
        self.max_len = max_len

    def forward(self, z, sos_token_id, targets=None, generate=False, teacher_forcing_ratio=0.5):
        batch_size = z.size(0)
        device = z.device

        hidden = (z.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1),
                  torch.zeros_like(z.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)))

        inputs = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)

        outputs = []
        hidden_states = []  # guardaremos todas las salidas de la LSTM aquí para atención

        if generate:
            for _ in range(self.max_len):
                embedded = self.embedding(inputs)
                output, hidden = self.lstm(embedded, hidden)
                hidden_states.append(output)

                context = self._compute_attention(output, hidden_states)
                concat_output = torch.cat([output, context], dim=-1)

                logits = self.fc_out(concat_output)
                outputs.append(logits)

                inputs = torch.argmax(logits, dim=-1)
        else:
            seq_len = targets.size(1)
            for t in range(seq_len):
                embedded = self.embedding(inputs)
                output, hidden = self.lstm(embedded, hidden)
                hidden_states.append(output)

                context = self._compute_attention(output, hidden_states)
                concat_output = torch.cat([output, context], dim=-1)

                logits = self.fc_out(concat_output)
                outputs.append(logits)

                if random.random() < teacher_forcing_ratio:
                    inputs = targets[:, t].unsqueeze(1)
                else:
                    inputs = torch.argmax(logits, dim=-1)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def _compute_attention(self, current_output, hidden_states):
        """
        Cálculo sencillo de atención: compara current_output contra todos los hidden_states anteriores.
        """
        # hidden_states: list de [batch, 1, hidden_dim]
        hidden_states_tensor = torch.cat(hidden_states, dim=1)  # [batch, time, hidden_dim]
        scores = torch.bmm(self.attention(current_output), hidden_states_tensor.transpose(1, 2))  # [batch, 1, time]
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, 1, time]
        context = torch.bmm(attn_weights, hidden_states_tensor)  # [batch, 1, hidden_dim]
        return context
