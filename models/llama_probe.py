from typing import Any, Optional

import torch
from torch import nn
from transformers import LlamaModel, LlamaConfig

from models import ChessLMProbe
from models.probes import get_probe


class ChessLLaMAProbe(ChessLMProbe):
    def __init__(self,
                 vocab_size: int = 77,
                 n_embd: int = 768,
                 n_positions: int = 1024,
                 n_layer: int = 12,
                 n_head: int = 12,
                 init_lr: float = 0.0003,
                 probe: nn.Module = None,
                 num_training_steps: Optional[int] = None,
                 *args, **kwargs
    ) -> None:
        super().__init__(init_lr=init_lr,
                         num_training_steps=num_training_steps,
                         probe=probe,
                         *args, **kwargs)

        config = LlamaConfig(vocab_size=vocab_size,
                             hidden_size=n_embd,
                             intermediate_size=int(2.6875*n_embd),
                             num_hidden_layers=n_layer,
                             num_attention_heads=n_head,
                             max_position_embeddings=n_positions,
                             bos_token_id=1,
                             eos_token_id=2,
                             pad_token_id=0)

        self.model = LlamaModel(config)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)


def get_model(model_type: str,
              vocab_size: int = 77,
              init_lr: float = 3e-4,
              num_training_steps: Optional[int] = None,
              checkpoint_path: Optional[str] = None,
              *args, **kwargs) -> ChessLLaMAProbe:

    if model_type == "chessllama_tiny_probe":
        n_embd = 192
        n_layer = 12
        n_head = 3
    if model_type == "chessllama_small_probe":
        n_embd = 384
        n_layer = 12
        n_head = 6
    if model_type == "chessllama_medium_probe":
        n_embd = 512
        n_layer = 12
        n_head = 8
    if model_type == "chessllama_base_probe":
        n_embd = 768
        n_layer = 12
        n_head = 12
    if model_type == "chessllama_large_probe":
        n_embd = 1024
        n_layer = 24
        n_head = 16

    probe = get_probe(n_dim=n_embd)

    if checkpoint_path is not None and checkpoint_path.endswith(".ckpt"):
        return ChessLLaMAProbe.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                    n_layer=n_layer,
                                                    n_embd=n_embd,
                                                    n_head=n_head,
                                                    vocab_size=vocab_size,
                                                    init_lr=init_lr,
                                                    probe=probe,
                                                    num_training_steps=num_training_steps,
                                                    map_location="cpu")
    else:
        model = ChessLLaMAProbe(vocab_size=vocab_size,
                                n_embd=n_embd,
                                n_layer=n_layer,
                                n_head=n_head,
                                init_lr=init_lr,
                                probe=probe,
                                num_training_steps=num_training_steps)

        if checkpoint_path is not None:
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        return model
