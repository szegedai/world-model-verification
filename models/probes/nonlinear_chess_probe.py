import torch
from torch import nn


class NonLinearChessProbe(nn.Module):
    def __init__(self, n_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_features=n_dim, out_features=n_dim, bias=True)
        self.relu = nn.ReLU()
        self.head = nn.Linear(in_features=n_dim, out_features=13*64, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.relu(self.fc1(x)))
        logits = logits.reshape(-1, 64, 13)
        return logits

