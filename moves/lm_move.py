from typing import List, Union

import torch
from chess import Board

from data_utils.chess_tokenizer import ChessTokenizer
from models import ChessLM
from utils import tokenize_sequence


def lm_move(board: Board,
            game_prefix: str,
            move_number: int,
            model: ChessLM,
            tokenizer: ChessTokenizer,
            device: Union[torch.device, str]
) -> str:
    """LM move adapter so that it conforms to `moves.MoveFunction`."""
    return chess_lm_move(model,
                         tokenizer,
                         tokenize_sequence(game_prefix.split(" "), tokenizer),
                         device)


def chess_lm_move(model: ChessLM,
                  tokenizer: ChessTokenizer,
                  game_prefix: List[int],
                  device: Union[torch.device, str]
) -> str:

    greedy_game_prefix = list(game_prefix)
    prefix_tens = torch.tensor([greedy_game_prefix]).to(device)
    pred_move = ""

    for idx in range(3):
        logits = model(prefix_tens)
        last_token_logit = logits[0, -1, :]
        token_idx = torch.argmax(last_token_logit).item()
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
