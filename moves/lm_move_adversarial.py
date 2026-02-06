from typing import List, Union, Iterable, Any, Tuple

import torch
from chess import Board

from data_utils.chess_tokenizer import ChessTokenizer
from models import ChessLM
from utils import tokenize_sequence, get_legal_moves


def lm_move_adversarial(board: Board,
                        game_prefix: str,
                        move_number: int,
                        model: ChessLM,
                        tokenizer: ChessTokenizer,
                        device: Union[torch.device, str]
) -> Tuple[str, Any]:
    """LM move adapter so that it conforms to `moves.MoveFunction`."""
    return chess_lm_move_adversarial(model,
                                     tokenizer,
                                     tokenize_sequence(game_prefix.split(" "), tokenizer),
                                     get_legal_moves(board),
                                     device)


def chess_lm_move_adversarial(model: ChessLM,
                              tokenizer: ChessTokenizer,
                              game_prefix: List[int],
                              legal_moves: Iterable[str],
                              device: Union[torch.device, str]
) -> Tuple[str, Any]:
    """Returns an LM move that is corrected for legality.

    The LM move returned by this function is guaranteed to be legal and has the
    highest prefix (first-token) probability.
    """

    greedy_game_prefix = list(game_prefix)
    prefix_tens = torch.tensor([greedy_game_prefix]).to(device)
    pred_move = ""

    for idx in range(3):
        logits = model(prefix_tens)
        last_token_logit = logits[0, -1, :]
        sorted_tokens = torch.argsort(last_token_logit, descending=False)

        ok = False
        for token in sorted_tokens:
            move_part = tokenizer.decode_token(token)
            # filter ambigous move prefixes
            # e.g. the decoded token 'b' stands for promotion to a bishop, but
            # it is also a prefix of a move like 'b2c3'
            if idx < 2 and len(move_part) != 2: continue
            if idx == 2 and len(move_part) != 1: continue # this is not strictly necessary

            pred_move_new = pred_move + move_part

            for legal_move in legal_moves:
                if legal_move.startswith(pred_move_new):
                    pred_move = pred_move_new
                    greedy_game_prefix += [token]
                    prefix_tens = torch.tensor([greedy_game_prefix]).to(device)
                    ok = True
                    break # new partial move is a legal move prefix

            if ok: break # found the top legal move prefix

        if idx == 1:
            need_third_round = False
            for legal_move in legal_moves:
                if legal_move.startswith(pred_move) and len(legal_move) > 4:
                    need_third_round = True

            if not need_third_round: break # no need to check promotion token

    return (pred_move, None)
