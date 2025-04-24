import torch.nn as nn

class CrazyMonkeyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, drop_prob=0.5):
        super().__init__(in_features, out_features, bias)
        self.drop_prob = drop_prob

    def forward(self, input):
        if self.training and self.drop_prob > 0:
            weight = nn.functional.dropout(self.weight, p=self.drop_prob, training=True)
        else:
            weight = self.weight
        return nn.functional.linear(input, weight, self.bias)
