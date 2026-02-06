from typing import Optional

from .chess_lm import ChessLM
from .chess_lm_probe import ChessLMProbe
from .gpt2 import ChessGPT2
from .llama import ChessLLaMA
from .probes import get_probe, setup_probe


def get_model(model_type: str,
              vocab_size: int = 77,
              init_lr: float = 3e-4,
              num_training_steps: Optional[int] = None,
              checkpoint_path: Optional[str] = None,
              *args, **kwargs
) -> ChessLM:
    if "probe" in model_type:
        if "chessgpt2" in model_type:
            from .gpt2_probe import get_model
            return get_model(model_type=model_type,
                             vocab_size=vocab_size,
                             init_lr=init_lr,
                             num_training_steps=num_training_steps,
                             checkpoint_path=checkpoint_path,
                             *args, **kwargs)
        elif "llama" in model_type:
            from .llama_probe import get_model
            return get_model(model_type=model_type,
                             vocab_size=vocab_size,
                             init_lr=init_lr,
                             num_training_steps=num_training_steps,
                             checkpoint_path=checkpoint_path,
                             *args, **kwargs)
        elif "mamba" in model_type:
            from .mamba_probe import get_model
            return get_model(model_type=model_type,
                             vocab_size=vocab_size,
                             init_lr=init_lr,
                             num_training_steps=num_training_steps,
                             checkpoint_path=checkpoint_path,
                             *args, **kwargs)
    else:
        if "chessgpt2" in model_type:
            from .gpt2 import get_model
            return get_model(model_type=model_type,
                             vocab_size=vocab_size,
                             init_lr=init_lr,
                             num_training_steps=num_training_steps,
                             checkpoint_path=checkpoint_path,
                             *args, **kwargs)
        elif "llama" in model_type:
            from .llama import get_model
            return get_model(model_type=model_type,
                             vocab_size=vocab_size,
                             init_lr=init_lr,
                             num_training_steps=num_training_steps,
                             checkpoint_path=checkpoint_path,
                             *args, **kwargs)
        elif "mamba" in model_type:
            from .mamba import get_model
            return get_model(model_type=model_type,
                             vocab_size=vocab_size,
                             init_lr=init_lr,
                             num_training_steps=num_training_steps,
                             checkpoint_path=checkpoint_path,
                             *args, **kwargs)
        elif "rwkv" in model_type:
            from .rwkv import get_model
            return get_model(model_type=model_type,
                             vocab_size=vocab_size,
                             init_lr=init_lr,
                             num_training_steps=num_training_steps,
                             checkpoint_path=checkpoint_path,
                             *args, **kwargs)
        elif "xlstm" in model_type:
            from .xlstm import get_model
            return get_model(model_type=model_type,
                             vocab_size=vocab_size,
                             init_lr=init_lr,
                             num_training_steps=num_training_steps,
                             checkpoint_path=checkpoint_path,
                             *args, **kwargs)
