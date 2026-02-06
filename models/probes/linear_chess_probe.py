import torch
from torch import nn


class LinearChessProbe(nn.Module):
    def __init__(self, n_dim: int, bias: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(in_features=n_dim, out_features=13*64, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        logits = logits.reshape(-1, 64, 13)
        return logits
