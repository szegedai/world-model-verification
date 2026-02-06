from typing import Any, Optional

import torch
from torch import nn
from transformers import RwkvModel, RwkvConfig
from lightning import LightningModule

from models import ChessLM


class RWKVHeadModel(LightningModule):
    """RWKVModel wrapper.

    This wrapper only serves to handle args and kwargs in the forward
    function that are not present in the original RWKVModel's
    forward implementation.
    """

    def __init__(self, config: RwkvConfig,
                 *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transformer = RwkvModel(config)
        self.head = nn.Linear(config.hidden_size,
                              config.vocab_size,
                              bias=False)

    def forward(self,
                x: torch.Tensor,
                *args, **kwargs
    ) -> torch.Tensor:
        x = self.transformer(x)[0]
        return self.head(x)


class ChessRWKV(ChessLM):
    def __init__(self,
                 vocab_size: int = 77,
                 n_embd: int = 768,
                 n_layer: int = 12,
                 init_lr: float = 0.0003,
                 num_training_steps: Optional[int] = None,
                 *args, **kwargs
    ) -> None:
        super().__init__(init_lr=init_lr,
                         num_training_steps=num_training_steps,
                         *args, **kwargs)

        config = RwkvConfig(vocab_size=vocab_size,
                            hidden_size=n_embd,
                            num_hidden_layers=n_layer,
                            pad_token_id=0,
                            bos_token_id=1,
                            eos_token_id=2)

        self.model = RWKVHeadModel(config)


def get_model(model_type: str,
              vocab_size: int = 77,
              init_lr: float = 3e-4,
              num_training_steps: Optional[int] = None,
              checkpoint_path: Optional[str] = None,
              *args, **kwargs) -> ChessRWKV:

    if model_type == "chessrwkv_tiny":
        n_embd = 192
        n_layer = 12
    if model_type == "chessrwkv_small":
        n_embd = 384
        n_layer = 12
    if model_type == "chessrwkv_medium":
        n_embd = 512
        n_layer = 12
    if model_type == "chessrwkv_base":
        n_embd = 768
        n_layer = 12
    if model_type == "chessrwkv_large":
        n_embd = 1024
        n_layer = 24

    if checkpoint_path is not None and checkpoint_path.endswith(".ckpt"):
        return ChessRWKV.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                              n_layer=n_layer,
                                              n_embd=n_embd,
                                              vocab_size=vocab_size,
                                              init_lr=init_lr,
                                              num_training_steps=num_training_steps,
                                              map_location="cpu")
    else:
        model = ChessRWKV(vocab_size=vocab_size,
                          n_embd=n_embd,
                          n_layer=n_layer,
                          init_lr=init_lr,
                          num_training_steps=num_training_steps)

        if checkpoint_path is not None:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        return model
