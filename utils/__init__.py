from typing import Set, List

import torch
from chess import Board
from torch import nn

from data_utils.chess_tokenizer import ChessTokenizer


def get_legal_moves(board: Board) -> Set[str]:
    legal_moves = list()
    for move in board.legal_moves:
        uci_move = board.uci(move)
        legal_moves.append(uci_move)

    return legal_moves


def tokenize_sequence(game_sequence: List[str],
                      tokenizer: ChessTokenizer
) -> List[int]:

    tokenized_sequence = [tokenizer.bos_token_id]

    for move in game_sequence:
        tokenized_sequence.extend(tokenizer.encode(move, False, False))

    return tokenized_sequence


def freeze(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer
    return None


def get_internal_activation(data: torch.Tensor,
                            model: nn.Module,
                            layer_name: str,
) -> torch.Tensor:
    """"""

    layer = get_layer(model, layer_name)
    activation = None

    def act_store_hook(m, i, o):
        nonlocal activation
        if type(o) == tuple:
            activation = o[0]
        else:
            activation = o


    hook_handler = layer.register_forward_hook(act_store_hook)
    model(data)
    hook_handler.remove()

    return activation

