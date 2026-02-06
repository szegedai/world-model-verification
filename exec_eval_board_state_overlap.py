import math
import os
import sys
import json
from typing import Optional
from argparse import ArgumentParser
from collections import defaultdict

import chess
import torch
import numpy as np
from lightning import seed_everything
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.line_dataset import LineByLineTextDataset
from data_utils.data_collator import DataCollatorForLanguageModeling
from data_utils.chess_tokenizer import ChessTokenizer
from models import get_model, setup_probe
from utils import freeze, get_legal_moves
from utils.probing import convert_fen_to_labels, get_activations_with_fens
from train_utils import probe_loss_and_acc_stats


LABEL_PIECES = {
    1: 'P',
    2: 'p',
    3: 'R',
    4: 'r',
    5: 'N',
    6: 'n',
    7: 'B',
    8: 'b',
    9: 'Q',
    10: 'q',
    11: 'K',
    12: 'k',
}


def decoder_output_to_board_clever(dec_out: torch.Tensor,
                                   white_to_move: bool
) -> chess.Board:
    """`dec_out` is a 2D tensor of shape 64x13.

    This function assumes global coding, not mine-yours, i.e. `K` is label 11,
    `k` is label 12. This approach ensures that both sides have exactly one
    king, however, this makes it so that the decoded board is less faithful to
    the decoder's raw output.
    """
    dec_out = torch.softmax(dec_out, 1)

    # Ensure that only 1 king of each color will be placed on the board
    K_idx = torch.argmax(dec_out[:, 11]).item()
    k_idx = torch.argmax(dec_out[:, 12]).item()
    dec_out[:, 11] = 0
    dec_out[:, 12] = 0

    # Decode the board state
    piece_map = {}
    for i in range(64):
        if i == K_idx: piece_map[i] = chess.Piece.from_symbol("K")
        elif i == k_idx: piece_map[i] = chess.Piece.from_symbol("k")
        else:
            piece_idx = torch.argmax(dec_out[i]).item()
            if piece_idx > 0:
                piece_map[i] = chess.Piece.from_symbol(LABEL_PIECES[piece_idx])

    board = chess.Board(None)
    board.set_piece_map(piece_map)
    board.turn = chess.WHITE if white_to_move else chess.BLACK
    return board


def decoder_output_to_board_naive(dec_out: torch.Tensor,
                                  white_to_move: bool
) -> chess.Board:
    """`dec_out` is a 2D tensor of shape 64x13.

    This function assumes global coding, not mine-yours, i.e. `K` is label 11,
    `k` is label 12. This is a naive approach that will likely result in an
    illegal board if the decoding is bad.
    """
    dec_out = torch.softmax(dec_out, 1)

    # Decode the board state
    piece_map = {}
    for i in range(64):
        piece_idx = torch.argmax(dec_out[i]).item()
        if piece_idx > 0:
            piece_map[i] = chess.Piece.from_symbol(LABEL_PIECES[piece_idx])

    board = chess.Board(None)
    board.set_piece_map(piece_map)
    board.turn = chess.WHITE if white_to_move else chess.BLACK
    return board


def _parse_args(args):
    parser = ArgumentParser("Chess LLM legal moves overlap eval")
    # Model
    parser.add_argument("-m", "--model", type=str, default="chessgpt2_base", dest="model_type")
    parser.add_argument("-mp", "--model-path", type=str, required=True, dest="model_path")
    parser.add_argument("-pp", "--probe-path", type=str, required=True, dest="probe_path")
    parser.add_argument("-l", "--layer", type=str, default="model.transformer.h.11", dest="layer_name")

    # Dataset access
    parser.add_argument("--vocab-file", type=str, dest="vocab_file")
    parser.add_argument("--data-file", type=str, dest="data_file")

    # Hyperparams
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-3, dest="epsilon")
    parser.add_argument("-rap", "--rap-prob", type=float, default=0.0, dest="rap_prob")
    parser.add_argument("-clever", "--clever-decoding", action="store_true", default=False, dest="clever_decoding")
    parser.add_argument("-gpu", "--gpu-id", type=int, default=0, dest="gpu")

    # Save folder
    parser.add_argument("-sf", "--save-folder", type=str, default="./overlaps", dest="save_folder")

    return parser.parse_args(args)


def main(conf):

    # epsilon = 0.001
    batch_size = 1
    # clever_decoding = False
    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"
    # save_folder = "overlaps/chessgpt2_tiny_rand500k_probs"

    # Model and probe
    model = get_model(conf.model_type, checkpoint_path=conf.model_path)
    model.to(device)
    model.eval()
    joint_probe = "probe" in conf.model_type
    if joint_probe:
        probe = nn.Identity() # placeholder
    else:
        probe = setup_probe(model,
                            conf.layer_name,
                            device,
                            load_from_file=conf.probe_path)
    probe.to(device)
    probe.eval()

    # Dataset
    max_positions = 800
    tokenizer = ChessTokenizer(conf.vocab_file)
    dataset = LineByLineTextDataset(tokenizer,
                                    conf.data_file,
                                    max_positions,
                                    rap_prob=conf.rap_prob)

    collate_fn = DataCollatorForLanguageModeling(tokenizer)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)

    gt_mp_ious = defaultdict(list)
    gt_bs_ious = defaultdict(list)
    mp_bs_ious = defaultdict(list)
    all_ious = defaultdict(list)
    gt_mp_iou_list = []
    gt_bs_iou_list = []
    mp_bs_iou_list = []
    all_iou_list = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            out = model(data['input_ids'].to(device))
            activations, fens = get_activations_with_fens(model,
                                                          tokenizer,
                                                          data,
                                                          conf.layer_name,
                                                          device,
                                                          joint_probe=joint_probe)
            probe_out = probe(activations.to(device)).detach().cpu()
            out = out.detach().cpu()
            out = out[data['separator_ind'] == 1]

            # Possible starting positions based on model prediction
            mask = torch.softmax(out, 1) > conf.epsilon
            legal_tokens_per_move = [torch.nonzero(m).squeeze(1) for m in mask]
            mp_squares_per_move = [
                set([tokenizer.decode_token(t) for t in legal_tokens])
                for legal_tokens in legal_tokens_per_move
            ]

            # Ground truth positions
            gt_squares_per_move = []
            for fen in fens:
                board = chess.Board(fen)
                moves = get_legal_moves(board)
                gt_squares_per_move.append(set([move[:2] for move in moves]))

            # Possible starting positions based on board state decoding
            bs_squares_per_move = []
            for i, dec_out in enumerate(probe_out):
                white_to_move = (i % 2 == 0)
                if conf.clever_decoding:
                    board = decoder_output_to_board_clever(dec_out, white_to_move)
                else:
                    board = decoder_output_to_board_naive(dec_out, white_to_move)
                moves = get_legal_moves(board)
                bs_squares_per_move.append(set([move[:2] for move in moves]))

            for i, (gt, mp, bs) in enumerate(zip(gt_squares_per_move, mp_squares_per_move, bs_squares_per_move)):
                # print(f"{i}:")
                if len(gt.union(mp)) != 0:
                    iou_gt_mp = len(gt.intersection(mp)) / len(gt.union(mp))
                    gt_mp_ious[i].append(iou_gt_mp)
                    gt_mp_iou_list.append(iou_gt_mp)

                if len(gt.union(bs)) != 0:
                    iou_gt_bs = len(gt.intersection(bs)) / len(gt.union(bs))
                    gt_bs_ious[i].append(iou_gt_bs)
                    gt_bs_iou_list.append(iou_gt_bs)

                if len(mp.union(bs)) != 0:
                    iou_mp_bs = len(mp.intersection(bs)) / len(mp.union(bs))
                    mp_bs_ious[i].append(iou_mp_bs)
                    mp_bs_iou_list.append(iou_mp_bs)

                if len(mp.union(bs).union(gt)) != 0:
                    iou_all = len(mp.intersection(bs).intersection(gt)) / len(mp.union(bs).union(gt))
                    all_ious[i].append(iou_all)
                    all_iou_list.append(iou_all)

    # save outputs
    os.makedirs(conf.save_folder, exist_ok=True)
    with open(os.path.join(conf.save_folder, "gt_mp_ious.json"), "w+") as f:
        json.dump(gt_mp_ious, f)
    with open(os.path.join(conf.save_folder, "gt_bs_ious.json"), "w+") as f:
        json.dump(gt_bs_ious, f)
    with open(os.path.join(conf.save_folder, "mp_bs_ious.json"), "w+") as f:
        json.dump(mp_bs_ious, f)
    with open(os.path.join(conf.save_folder, "all_ious.json"), "w+") as f:
        json.dump(all_ious, f)
    with open(os.path.join(conf.save_folder, "mean_ious.json"), "w+") as f:
        json.dump({
            'gt_mp': np.mean(np.array(gt_mp_iou_list)),
            'gt_bs': np.mean(np.array(gt_bs_iou_list)),
            'mp_bs': np.mean(np.array(mp_bs_iou_list)),
            'all': np.mean(np.array(all_iou_list)),
        }, f)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(_parse_args(args))
