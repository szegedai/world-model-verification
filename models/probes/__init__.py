from typing import Optional

import torch
from torch import nn

from utils import get_internal_activation

from .conv1d_chess_probe import Conv1dChessProbe
from .linear_chess_probe import LinearChessProbe
from .nonlinear_chess_probe import NonLinearChessProbe


def get_probe(n_dim: int,
              linear: bool = True,
              n_tokens_per_position: int = 1,
              *args, **kwargs) -> nn.Module:
    if n_tokens_per_position == 1:
        if linear:
            return LinearChessProbe(n_dim, *args, **kwargs)
        else:
            return NonLinearChessProbe(n_dim, *args, **kwargs)
    else:
        return Conv1dChessProbe(n_dim, n_tokens_per_position)


def setup_probe(model: nn.Module,
                layer_name: nn.Module,
                device: torch.device,
                linear: bool = True,
                load_from_file: Optional[str] = None,
                n_tokens_per_position: int = 1,
                *args, **kwargs
) -> nn.Module:
    # Put model in eval mode
    model.to(device)
    model.eval()

    # Get embedding dimension using a fake input
    x_fake = torch.tensor([[1]]).to(device)
    act_fake = get_internal_activation(x_fake, model, layer_name)
    n_dim = act_fake.shape[2]

    # Setup and load probe
    probe = get_probe(n_dim, linear, n_tokens_per_position, *args, **kwargs)

    if load_from_file:
        print(f"Loading from file: {load_from_file}")
        probe.load_state_dict(torch.load(load_from_file, map_location="cpu"))

    probe.to(device)
    probe.eval()

    return probe
