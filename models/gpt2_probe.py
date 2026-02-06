from typing import Optional

import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

from models import ChessLMProbe
from models.probes import get_probe


class ChessGPT2Probe(ChessLMProbe):
    def __init__(self,
                 vocab_size: int = 77,
                 n_embd: int = 768,
                 n_positions: int = 1024,
                 n_layer: int = 12,
                 n_head: int = 12,
                 init_lr: float = 3e-4,
                 probe: nn.Module = None,
                 num_training_steps: Optional[int] = None,
                 *args, **kwargs
    ) -> None:
        # Super lazy solution due to time constraints
        assert probe is not None, "Probe cannot be None!"

        super().__init__(init_lr=init_lr,
                         num_training_steps=num_training_steps,
                         probe=probe,
                         *args, **kwargs)

        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_positions=n_positions,
            n_layer=n_layer,
            n_head=n_head
        )

        self.model = GPT2Model(config)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)


def get_model(model_type: str,
              vocab_size: int = 77,
              init_lr: float = 3e-4,
              num_training_steps: Optional[int] = None,
              checkpoint_path: Optional[str] = None,
              *args, **kwargs
) -> ChessGPT2Probe:

    if model_type == "chessgpt2_tiny_probe":
        n_embd = 192
        n_layer = 12
        n_head = 3
    if model_type == "chessgpt2_small_probe":
        n_embd = 384
        n_layer = 12
        n_head = 6
    if model_type == "chessgpt2_medium_probe":
        n_embd = 512
        n_layer = 12
        n_head = 8
    if model_type == "chessgpt2_base_probe":
        n_embd = 768
        n_layer = 12
        n_head = 12
    if model_type == "chessgpt2_large_probe":
        n_embd = 1024
        n_layer = 24
        n_head = 16

    probe = get_probe(n_dim=n_embd)
    if checkpoint_path is not None and checkpoint_path.endswith(".ckpt"):
        return ChessGPT2Probe.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                   n_layer=n_layer,
                                                   n_embd=n_embd,
                                                   n_head=n_head,
                                                   vocab_size=vocab_size,
                                                   init_lr=init_lr,
                                                   probe=probe,
                                                   num_training_steps=num_training_steps,
                                                   map_location="cpu")
    else:
        model = ChessGPT2Probe(vocab_size=vocab_size,
                               n_embd=n_embd,
                               n_layer=n_layer,
                               n_head=n_head,
                               init_lr=init_lr,
                               probe=probe,
                               num_training_steps=num_training_steps)

        if checkpoint_path is not None:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        return model
