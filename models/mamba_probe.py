from typing import Any, Optional

import torch
from torch import nn
from transformers import MambaModel, MambaConfig

from models import ChessLMProbe
from models.probes import get_probe


class ChessMambaProbe(ChessLMProbe):
    def __init__(self,
                 vocab_size: int = 77,
                 n_embd: int = 768,
                 n_layer: int = 12,
                 init_lr: float = 0.0003,
                 probe: nn.Module = None,
                 num_training_steps: Optional[int] = None,
                 *args, **kwargs
    ) -> None:
        super().__init__(init_lr=init_lr,
                         num_training_steps=num_training_steps,
                         probe=probe,
                         *args, **kwargs)

        config = MambaConfig(vocab_size=vocab_size,
                             hidden_size=n_embd,
                             state_size=16,
                             num_hidden_layers=n_layer,
                             pad_token_id=0,
                             bos_token_id=1,
                             eos_token_id=2)

        self.model = MambaModel(config)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)


def get_model(model_type: str,
              vocab_size: int = 77,
              init_lr: float = 3e-4,
              num_training_steps: Optional[int] = None,
              checkpoint_path: Optional[str] = None,
              *args, **kwargs) -> ChessMambaProbe:

    if model_type == "chessmamba_tiny_probe":
        n_embd = 192
        n_layer = 12
    if model_type == "chessmamba_small_probe":
        n_embd = 384
        n_layer = 12
    if model_type == "chessmamba_medium_probe":
        n_embd = 512
        n_layer = 12
    if model_type == "chessmamba_base_probe":
        n_embd = 768
        n_layer = 12
    if model_type == "chessmamba_large_probe":
        n_embd = 1024
        n_layer = 24

    probe = get_probe(n_dim=n_embd)

    if checkpoint_path is not None and checkpoint_path.endswith(".ckpt"):
        return ChessMambaProbe.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                    n_layer=n_layer,
                                                    n_embd=n_embd,
                                                    vocab_size=vocab_size,
                                                    init_lr=init_lr,
                                                    num_training_steps=num_training_steps,
                                                    probe=probe,
                                                    map_location="cpu")
    else:
        model = ChessMambaProbe(vocab_size=vocab_size,
                                n_embd=n_embd,
                                n_layer=n_layer,
                                init_lr=init_lr,
                                probe=probe,
                                num_training_steps=num_training_steps)

        if checkpoint_path is not None:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        return model
