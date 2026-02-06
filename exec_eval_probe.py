import os
import sys
import json
from argparse import ArgumentParser
from collections import defaultdict

import torch
import numpy as np
from lightning import seed_everything
from torch import nn
from tqdm import tqdm

from data_utils.line_dataset import LineByLineTextDataset
from data_utils.data_collator import DataCollatorForLanguageModeling
from data_utils.chess_tokenizer import ChessTokenizer
from models import get_model, setup_probe
from utils import freeze
from utils.probing import convert_fen_to_labels, get_activations_with_fens


def _parse_args(args):
    parser = ArgumentParser("Board state probe training")

    # Model
    parser.add_argument("-m", "--model", type=str, default="chessgpt2_base", dest="model_type")
    parser.add_argument("-mp", "--model-path", type=str, required=True, dest="model_path")
    parser.add_argument("-l", "--layer", type=str, default="model.transformer.h.11", dest="layer_name")

    # Probe
    parser.add_argument("-pt", "--probe-type", type=str, default="linear", choices=["linear", "nonlinear"], dest="probe_type")
    parser.add_argument("-pp", "--probe_path", type=str, dest="probe_path")
    parser.add_argument("--mine-yours", action="store_true", default=False, dest="mine_yours")
    parser.add_argument("--mask-positions", type=str, default=None, choices=[None, "only_black", "only_white"], dest="mask_positions")
    parser.add_argument("--n-tokens-per-position", type=int, default=1, dest="n_tokens_per_position")

    # Dataset
    parser.add_argument("--vocab-file", type=str, dest="vocab_file")
    parser.add_argument("--data-file", type=str,dest="data_file")
    parser.add_argument("--rap-prob", type=float, default=0.0, dest="rap_prob")

    # General
    parser.add_argument("-sf", "--save-folder", type=str, default="./probe_evals", dest="save_folder")
    parser.add_argument("-gpu", "--gpu-id", type=int, default=0, dest="gpu")
    parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")

    return parser.parse_args(args)


def acc_stats_per_move(preds, labels):
    assert preds.shape[0] == labels.shape[0]

    acc_per_move = []
    piece_acc_per_move = []
    n_corr = 0
    n_piece_corr = 0
    n_all = 0
    n_piece_all = 0

    for i in range(len(preds)):
        pred = torch.argmax(preds[i], 1)
        label = torch.argmax(labels[i], 1)
        piece_label = label[(label + pred) > 0]
        piece_pred = pred[(label + pred) > 0]

        corr = torch.sum(pred == label).item()
        piece_corr = torch.sum(piece_pred == piece_label).item()

        acc_per_move.append(corr / len(pred))
        piece_acc_per_move.append(piece_corr / len(piece_pred))

        n_corr += corr
        n_piece_corr += piece_corr
        n_all += len(pred)
        n_piece_all += len(piece_pred)

    return acc_per_move, piece_acc_per_move, n_corr, n_all, n_piece_corr, n_piece_all




def main(conf):
    seed_everything(conf.seed)
    # Model
    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"
    model = get_model(conf.model_type, checkpoint_path=conf.model_path)
    model.eval()
    freeze(model)
    model.to(device)
    layer_name = conf.layer_name
    joint_probe = "probe" in conf.model_type

    # Dataset
    max_positions = 800
    tokenizer = ChessTokenizer(conf.vocab_file)
    dataset = LineByLineTextDataset(tokenizer,
                                    conf.data_file,
                                    max_positions,
                                    rap_prob=conf.rap_prob)

    collate_fn = DataCollatorForLanguageModeling(tokenizer)

    data_loader = torch.utils.data.DataLoader(dataset, 1, False, collate_fn=collate_fn)

    # Probe
    if joint_probe:
        probe = nn.Identity()
        n_tokens_per_position = 1
    else:
        n_tokens_per_position = conf.n_tokens_per_position
        linear = conf.probe_type == "linear"
        probe = setup_probe(model,
                            layer_name,
                            device,
                            linear=linear,
                            n_tokens_per_position=n_tokens_per_position,
                            load_from_file=conf.probe_path)
    probe.to(device)
    mine_yours = conf.mine_yours

    # loss_fn = nn.CrossEntropyLoss()
    accs = defaultdict(list)
    paccs = defaultdict(list)
    n_corr = 0
    n_all = 0
    n_piece_corr = 0
    n_piece_all = 0
    game_counter = 0

    for data in tqdm(data_loader):
        data = {
            "input_ids": data["input_ids"],
            "separator_ind": data["separator_ind"]
        }

        activations, fens = get_activations_with_fens(model,
                                                      tokenizer,
                                                      data,
                                                      layer_name,
                                                      device,
                                                      n_tokens_per_position,
                                                      joint_probe=joint_probe)

        # activation_batch = torch.stack(activation_batch).to(device)
        activations = activations.to(device)
        labels = [convert_fen_to_labels(fen, mine_yours=mine_yours)
                    for fen in fens]
        labels = torch.stack(labels).to(device)

        probe_out = probe(activations)

        acc_stats = acc_stats_per_move(probe_out, labels)
        acc_per_move, piece_acc_per_move, n_c, n_a, n_p_c, n_p_a = acc_stats
        n_corr += n_c
        n_all += n_a
        n_piece_corr += n_p_c
        n_piece_all += n_p_a
        for i in range(len(acc_per_move)):
            accs[i].append(acc_per_move[i])
            paccs[i].append(piece_acc_per_move[i])

        game_counter += len(data["input_ids"])
        if game_counter >= 15000:
            break

    move_mean_accs = []
    move_std_accs = []
    move_mean_paccs = []
    move_std_paccs = []

    for i in range(0, max(accs.keys())):
        move_accuracies = np.array(accs[i])
        move_piece_accuracies = np.array(paccs[i])

        move_mean_accs.append(np.mean(move_accuracies))
        move_std_accs.append(np.std(move_accuracies))
        move_mean_paccs.append(np.mean(move_piece_accuracies))
        move_std_paccs.append(np.std(move_piece_accuracies))

    # print("{")
    # print(f"'move_mean_accs': {move_mean_accs},")
    # print(f"'move_std_accs': {move_std_accs},")
    # print(f"'move_mean_paccs': {move_mean_paccs},")
    # print(f"'move_std_paccs': {move_std_paccs}")
    # print("}")

    # save outputs:
    os.makedirs(conf.save_folder, exist_ok=True)
    with open(os.path.join(conf.save_folder, "acc_eval.json"), "w+") as f:
        json.dump({
            'accuracy': n_corr / n_all,
            'piece_accuracy': n_piece_corr / n_piece_all,
            'move_mean_accs': move_mean_accs,
            'move_std_accs': move_std_accs,
            'move_mean_paccs': move_mean_paccs,
            'move_std_paccs': move_std_paccs,
        }, f)

    print(f"Accuracy: {n_corr / n_all}")
    print(f"Piece accuracy: {n_piece_corr / n_piece_all}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(_parse_args(args))

