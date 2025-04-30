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

        hidden = (
            z.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1),
            torch.zeros_like(z.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1))
        )

        inputs = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)

        if generate:
            seq_len = self.max_len
        else:
            seq_len = targets.size(1)

        outputs = []
        hidden_states = []

        for t in range(seq_len):
            embedded = self.embedding(inputs)
            output, hidden = self.lstm(embedded, hidden)  # output: [batch, 1, hidden_dim]
            hidden_states.append(output)

            # construye el tensor hasta el paso t
            past_hidden_states = torch.cat(hidden_states, dim=1)  # [batch, t+1, hidden_dim]
            context = self._compute_attention(output, past_hidden_states)
            concat_output = torch.cat([output, context], dim=-1)

            logits = self.fc_out(concat_output)
            outputs.append(logits)

            if generate:
                inputs = torch.argmax(logits, dim=-1)
            else:
                if random.random() < teacher_forcing_ratio:
                    inputs = targets[:, t].unsqueeze(1)
                else:
                    inputs = torch.argmax(logits, dim=-1)

        return torch.cat(outputs, dim=1)  # [batch, seq_len, vocab_size]

    def _compute_attention(self, current_output, past_hidden_states):
        """
        Atención sobre todos los hidden states pasados hasta ahora.
        current_output: [batch, 1, hidden_dim]
        past_hidden_states: [batch, t, hidden_dim]
        """
        scores = torch.bmm(self.attention(current_output), past_hidden_states.transpose(1, 2))  # [batch, 1, t]
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, 1, t]
        context = torch.bmm(attn_weights, past_hidden_states)  # [batch, 1, hidden_dim]
        return context
