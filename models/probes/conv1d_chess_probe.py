import torch
from torch import nn


class Conv1dChessProbe(nn.Module):
    def __init__(self, n_dim: int, kernel_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv1d(in_channels=n_dim,
                              out_channels=n_dim,
                              kernel_size=kernel_size,
                              padding="valid")
        self.head = nn.Linear(in_features=n_dim, out_features=64 * 13)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        x = torch.flatten(x.transpose(2, 1), start_dim=1)
        x = self.head(x)
        x = x.reshape(-1, 64, 13)
        return x
