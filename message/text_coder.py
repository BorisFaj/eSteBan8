import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextCompressor(nn.Module):
    def __init__(self, output_dim=64, pooling='cls', freeze_bert=True):
        """
        :param output_dim: tamaño final del vector comprimido
        :param pooling: 'cls' o 'mean' para extraer el embedding del texto
        :param freeze_bert: si True, no entrena DistilBERT (más rápido)
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.pooling = pooling
        self.reductor = nn.Linear(768, output_dim)

    def forward(self, texts):
        """
        :param texts: lista de strings (batch) o string único
        :return: tensor (batch_size, output_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        outputs = self.bert(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)

        if self.pooling == 'mean':
            pooled = hidden_states.mean(dim=1)  # average pooling
        else:
            pooled = hidden_states[:, 0, :]  # CLS token

        reduced = self.reductor(pooled)
        return reduced  # (batch_size, output_dim)
