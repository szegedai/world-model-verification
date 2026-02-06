from typing import List, Union, Dict, Tuple
from copy import deepcopy

import chess
import torch
from chess import Board
from torch import nn

from data_utils.chess_tokenizer import ChessTokenizer
from models import ChessLM

from utils import get_legal_moves
from train_utils import probe_loss_and_acc_stats
from utils.probing import get_activations_with_fens, convert_fen_to_labels


def _probe_cls_loss(loss_and_acc_stats):
    return loss_and_acc_stats[0].item()


def _probe_accuracy(loss_and_acc_stats):
    return 1 - (loss_and_acc_stats[1] / loss_and_acc_stats[2])


def _probe_piece_accuracy(loss_and_acc_stats):
    return 1 - (loss_and_acc_stats[3] / loss_and_acc_stats[4])


def search_for_probe_error(game_sequence: List[str],
                           model: ChessLM,
                           tokenizer: ChessTokenizer,
                           layer_name: str,
                           probe: nn.Module,
                           device: torch.device,
                           metric: str,
                           joint_probe: bool = False
) -> str:
    board = chess.Board()
    for move in game_sequence:
        board.push_uci(move)

    legal_moves = get_legal_moves(board)
    top_error = 0 - 1e-7
    top_move = None

    # Setup metric function
    if metric == "cls_loss": metric_fn = _probe_cls_loss
    elif metric == "accuracy": metric_fn = _probe_accuracy
    elif metric == "piece_accuracy": metric_fn = _probe_piece_accuracy

    for move in legal_moves:
        new_game_sequence = [*game_sequence, move]
        tokenized_game = tokenizer.encode(" ".join(new_game_sequence))

        data = {
            "input_ids": torch.tensor([tokenized_game[0]]),
            "separator_ind": [torch.tensor(v) for v in tokenized_game[1]]
        }

        board_copy = deepcopy(board)
        board_copy.push_uci(move)

        activations, fens = get_activations_with_fens(model,
                                                      tokenizer,
                                                      data,
                                                      layer_name,
                                                      device,
                                                      joint_probe=joint_probe)

        activations = activations.to(device)
        if joint_probe:
            probe_out = activations[-1].reshape(1, 64, 13)
        else:
            probe_out = probe(activations[-1].reshape((1, activations[-1].shape[-1])))
        label = convert_fen_to_labels(fens[-1]).reshape(1, 64, 13)
        label = label.to(device)

        stats = probe_loss_and_acc_stats(probe_out, label, torch.nn.CrossEntropyLoss())
        metric_value = metric_fn(stats)

        if metric_value > top_error:
            top_error = metric_value
            top_error_move = move

    return (top_error_move, top_error)


def oracle_probe_error(board: Board,
                       game_prefix: str,
                       move_number: int,
                       model: ChessLM,
                       tokenizer: ChessTokenizer,
                       layer_name: str,
                       probe: nn.Module,
                       device: torch.device,
                       metric: str,
                       joint_probe: bool = False
) -> str:
    """Oracle (probe error) adapter so that it conforms to `moves.MoveFunction`."""
    game_prefix_filtered = [s for s in game_prefix.split(" ") if len(s) > 0]
    return search_for_probe_error(game_prefix_filtered,
                                  model,
                                  tokenizer,
                                  layer_name,
                                  probe,
                                  device,
                                  metric,
                                  joint_probe)
