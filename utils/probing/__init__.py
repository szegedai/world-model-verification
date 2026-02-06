from typing import Dict, List, Union, Tuple

import chess
import torch
import numpy as np

from data_utils.chess_tokenizer import ChessTokenizer
from models import ChessLM
from utils import get_internal_activation


def get_move_end_activations(activations, separator_inds):
    # move_end_inds = torch.where(torch.tensor(separator_inds) == 1)[0]
    move_end_inds = torch.where(separator_inds == 1)[0]
    return activations[move_end_inds]


def _remove_rap_annotation(move: str) -> str:
    """Removes RAP annotation from a move.

    E.g. converts the RAP-annotated move string 'Pe2e4' to the UCI move 'e2e4'.
    """
    if move.startswith(('P', 'K', 'Q', 'R', 'N', 'B')):
        return move[1:]
    return move


def convert_moves_to_fen(data: Dict[str, Union[torch.Tensor, List[int]]],
                         tokenizer: ChessTokenizer
) -> List[str]:
    board = chess.Board()
    fens = []
    move = ""

    for token, sep in zip(data["input_ids"], data["separator_ind"]):
        move_part = tokenizer.decode_token(token)

        if move_part == "</s>" or move_part == "</pad>":
            break

        move += move_part

        if sep == 1:
            # print(move)

            # Add sequence start here
            if move_part != "<s>":
                board.push_uci(_remove_rap_annotation(move))

            move = ""
            fens.append(board.fen())

    return fens


def get_activations_with_fens(model: ChessLM,
                              tokenizer: ChessTokenizer,
                              data: Dict[str, torch.Tensor],
                              layer_name: str,
                              device: Union[str, torch.device],
                              n_tokens_per_fen: int = 1,
                              joint_probe: bool = False
) -> Tuple[torch.Tensor, List[str]]:

    # This part is here purely for backwards compatibility. In practice, when
    # LineByLineTextDataset is used with DataCollatorForLanguageModeling (which
    # is more than recommended), the separators will be stored in a Tensor, so
    # the else branch is irrelevant.
    if isinstance(data['separator_ind'], torch.Tensor):
        separators = data['separator_ind']
    else:
        separators = torch.transpose(torch.vstack(data["separator_ind"]), 0, 1)

    tokenized_input = data["input_ids"].to(device)
    n_samples = len(data["input_ids"])

    if not joint_probe:
        batch_activations = get_internal_activation(tokenized_input, model, layer_name).detach().cpu()

    activations = []
    fens = []
    for i in range(n_samples):
        data_item = {
            "input_ids": data["input_ids"][i],
            "separator_ind": separators[i]
        }
        if not joint_probe:
            act = batch_activations[i]
            acts_game = get_move_end_activations(act, data_item["separator_ind"]).numpy()
        else:
            input_ids = data_item['input_ids']
            input_ids = input_ids.reshape(1, input_ids.shape[0]).to(device)
            separator_ind = data_item['separator_ind']
            separator_ind = separator_ind.reshape(1, separator_ind.shape[0]).to(device)

            # input_ids = torch.tensor([data_item['input_ids']]).to(device)
            # separators = torch.tensor([data_item['separator_ind']]).to(device)
            out = model(input_ids, separator_ind=separator_ind)
            acts_game = out[1].detach().cpu().numpy()
        fens_game = convert_moves_to_fen(data_item, tokenizer)

        if n_tokens_per_fen == 1:
            activations.extend(acts_game)
            fens.extend(fens_game)

        else:
            k = n_tokens_per_fen
            for i in range(k, len(acts_game) + 1):
                activations.append(acts_game[i - k : i])
                fens.append(fens_game[i - 1])

    return (torch.tensor(np.array(activations)), fens)


_PIECE_LABELS_WHITE_TO_MOVE = {
    'P': 1,
    'p': 2,
    'R': 3,
    'r': 4,
    'N': 5,
    'n': 6,
    'B': 7,
    'b': 8,
    'Q': 9,
    'q': 10,
    'K': 11,
    'k': 12,
}


_PIECE_LABELS_BLACK_TO_MOVE = {
    'p': 1,
    'P': 2,
    'r': 3,
    'R': 4,
    'n': 5,
    'N': 6,
    'b': 7,
    'B': 8,
    'q': 9,
    'Q': 10,
    'k': 11,
    'K': 12,
}


def convert_fen_to_labels(fen: str, mine_yours: bool = False) -> torch.Tensor:
    """For converting to global labels, leave `white_to_move` as True."""
    board = chess.Board(fen)
    white_to_move = board.turn == chess.WHITE
    encoded_board = torch.zeros((64, 13))

    piece_positions = set()
    for pos, piece in board.piece_map().items():

        if (not mine_yours) or (mine_yours and white_to_move):
            label = _PIECE_LABELS_WHITE_TO_MOVE[piece.symbol()]
        else:
            label = _PIECE_LABELS_BLACK_TO_MOVE[piece.symbol()]

        encoded_board[pos, label] = 1
        piece_positions.add(pos)

    for pos in range(64):
        if pos not in piece_positions:
            encoded_board[pos, 0] = 1

    return encoded_board
