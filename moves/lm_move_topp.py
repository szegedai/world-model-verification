from typing import List, Union

import torch
import torch.nn.functional as F
from chess import Board

from data_utils.chess_tokenizer import ChessTokenizer
from models import ChessLM
from utils import tokenize_sequence


def sample_topp(logits: torch.Tensor, p: float) -> int:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits.scatter_(-1, indices_to_remove, float('-inf'))
    probabilities = torch.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    return sampled_token.item()


def lm_move_topp(board: Board,
                 game_prefix: str,
                 move_number: int,
                 model: ChessLM,
                 tokenizer: ChessTokenizer,
                 device: Union[torch.device, str],
                 p: float = 1.0
) -> str:
    """LM move adapter so that it conforms to `moves.MoveFunction`."""
    return chess_lm_move_topp(model,
                         tokenizer,
                         tokenize_sequence(game_prefix.split(" "), tokenizer),
                         device,
                         p)


def chess_lm_move_topp(model: ChessLM,
                       tokenizer: ChessTokenizer,
                       game_prefix: List[int],
                       device: Union[torch.device, str],
                       p: float = 1.0
) -> str:

    greedy_game_prefix = list(game_prefix)
    prefix_tens = torch.tensor([greedy_game_prefix]).to(device)
    pred_move = ""

    for idx in range(3):
        logits = model(prefix_tens)
        last_token_logit = logits[0, -1, :]
        token_idx = sample_topp(last_token_logit, p)
        current_token = tokenizer.decode_token(token_idx)
        pred_move += current_token

        if idx == 0 and current_token == tokenizer.eos_token:
            return ("quit", None)

        if idx < 2 and current_token in [tokenizer.bos_token, tokenizer.pad_token]:
            return (f"error_invalid_token: {current_token}", None)

        greedy_game_prefix += [token_idx]
        prefix_tens = torch.tensor([greedy_game_prefix]).to(device)

        if pred_move == tokenizer.eos_token:
            return ("quit", None)

    if len(pred_move) > 4:
        if len(pred_move) == 5 and pred_move[-1] not in ['q', 'r', 'b', 'n']:
            return (f"error_promotion: {pred_move}", None)
        elif len(pred_move) > 5:
            pred_move = pred_move[:4]

    return (pred_move, None)
