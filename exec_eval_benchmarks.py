import os
import chess
import sys
import json
import torch
import random
from argparse import ArgumentParser
from transformers import GPT2LMHeadModel, AutoModel
import torch
from torch import nn
from typing import Union, List, Tuple
from tqdm import tqdm

from data_utils.chess_tokenizer import ChessTokenizer
from data_utils.line_dataset import LineByLineTextDataset
from data_utils.data_collator import DataCollatorForLanguageModeling
from models import ChessLM, get_model


def _parse_args(args):
    parser = ArgumentParser("Chess-specific benchmark evaluation.")

    parser.add_argument("model_path", type=str)
    parser.add_argument("model_type", type=str)
    parser.add_argument("--vocab-dir", type=str, dest="vocab_dir")
    parser.add_argument("--data-dir", type=str, dest="data_dir")
    parser.add_argument("-sf", "--save-folder", type=str, default="./probe_evals", dest="save_folder")
    parser.add_argument("-gpu", "--gpu-id", type=int, dest="gpu_id", default=0)

    return parser.parse_args(args)


def get_legal_moves(board: chess.Board):
    legal_moves = list()
    # legal_moves = set()
    for move in board.legal_moves:
        uci_move = board.uci(move)
        # legal_moves.add(uci_move)
        legal_moves.append(uci_move)

    return legal_moves


def check_datasets(checkmate_ds: str,
                   stalemate_ds: str,
                   double_check_ds: str
) -> bool:
    ok = True
    if not os.path.isfile(checkmate_ds):
        print(f"Checkmate dataset is missing!")
        print(f"Expected file {checkmate_ds}")
        ok = False
    if not os.path.isfile(stalemate_ds):
        print(f"Stalemate dataset is missing!")
        print(f"Expected file {stalemate_ds}")
        ok = False
    if not os.path.isfile(double_check_ds):
        print(f"Double Check dataset is missing!")
        print(f"Expected file {double_check_ds}")
        ok = False
    return ok


def tokenize_sequence(game_sequence: List[str],
                      tokenizer: ChessTokenizer
) -> List[int]:

    tokenized_sequence = [tokenizer.bos_token_id]

    for move in game_sequence:
        tokenized_sequence.extend(tokenizer.encode(move, False, False))

    return tokenized_sequence


def eval_game_end(model: ChessLM,
                  tokenizer: ChessTokenizer,
                  dataset_file: str,
                  device: Union[str, torch.device]
) -> float:
    # lines = []
    # with open(dataset_file, "r") as f:
    #     lines = f.readlines()

    dataset = LineByLineTextDataset(tokenizer, dataset_file, 800, rap_prob=0)
    data_collator = DataCollatorForLanguageModeling(tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset, 1, collate_fn=data_collator)

    n_correct = 0
    n_all = 0

    for data in tqdm(data_loader):
        # moves = line.split(" ")[:-1] # remove trailing '\n'
        # tokenized_game_prefix = tokenize_sequence(moves, tokenizer)
        tokenized_game_prefix = data["input_ids"].to(device)
        # tokenized_game_prefix = torch.tensor([tokenized_game_prefix]).to(device)
        pred = model(tokenized_game_prefix)
        pred_token = tokenizer.decode_token(torch.argmax(pred[0, -2, :]).item())
        n_pred_end_tokens = torch.sum(torch.argmax(pred[0, :-1, :], 1) == tokenizer.eos_token_id).item()

        if pred_token == tokenizer.eos_token and n_pred_end_tokens == 1:
            n_correct += 1
        n_all += 1

    return n_correct / n_all


def eval_king_move(model: ChessLM,
                   tokenizer: ChessTokenizer,
                   dataset_file: str,
                   device: Union[str, torch.device]
) -> Tuple[float, float]:
    lines = []
    with open(dataset_file, "r") as f:
        lines = f.readlines()

    n_correct_intention = 0
    n_correct = 0
    n_all = 0

    for line in tqdm(lines):
        n_all += 1
        moves = line.split(" ")[:-1] # remove trailing '\n'
        board = chess.Board()
        for move in moves: board.push_uci(move)


        # Check if the model intends to move the king
        tokenized_game_prefix = tokenize_sequence(moves, tokenizer)
        tokenized_game_prefix = torch.tensor([tokenized_game_prefix]).to(device)
        pred = model(tokenized_game_prefix)
        pred_token = tokenizer.decode_token(torch.argmax(pred[0, -1, :]).item())

        if pred_token == chess.square_name(board.king(board.turn)):
            n_correct_intention += 1
        else:
            continue

        # Check if the king move is legal
        model_move = pred_token
        legal_moves = get_legal_moves(board)

        tokenized_game_prefix = tokenize_sequence([*moves, pred_token], tokenizer)
        tokenized_game_prefix = torch.tensor([tokenized_game_prefix]).to(device)
        pred = model(tokenized_game_prefix)
        pred_token = tokenizer.decode_token(torch.argmax(pred[0, -1, :]).item())
        model_move += pred_token

        if model_move in legal_moves:
            n_correct += 1

    return (n_correct_intention / n_all, n_correct / n_all)


def main(conf):
    # Check the availability of datasets
    checkmate_ds = os.path.join(conf.data_dir, "checkmate.txt")
    checkmate_100_150_ds = os.path.join(conf.data_dir, "checkmate_100_150.txt")
    stalemate_ds = os.path.join(conf.data_dir, "stalemate.txt")
    double_check_ds = os.path.join(conf.data_dir, "double_check.txt")

    if not check_datasets(checkmate_ds, stalemate_ds, double_check_ds):
        sys.exit(1)

    device = f"cuda:{conf.gpu_id}" if conf.gpu_id >= 0 else "cpu"

    model = get_model(conf.model_type,
                      checkpoint_path=conf.model_path)
    model.to(device)
    model.eval()

    vocab_file = os.path.join(conf.vocab_dir, "vocab.txt")
    tokenizer = ChessTokenizer(vocab_file)

    checkmate_score = eval_game_end(model, tokenizer, checkmate_ds, device)
    checkmate_100_150_score = eval_game_end(model, tokenizer, checkmate_100_150_ds, device)
    stalemate_score = eval_game_end(model, tokenizer, stalemate_ds, device)
    dc_intention, dc_score = eval_king_move(model, tokenizer, double_check_ds, device)

    os.makedirs(conf.save_folder, exist_ok=True)
    save_file = os.path.join(conf.save_folder, "benchmark_eval.json")
    with open(save_file, "w+") as f:
        json.dump({
            "checkmate_lt100": checkmate_score,
            "checkmate_100_150": checkmate_100_150_score,
            "checkmate_full": (checkmate_score + checkmate_100_150_score) / 2,
            "stalemate": stalemate_score,
            "double_check_intention": dc_intention,
            "double_check_correctness": dc_score
        }, f)

    # print(f"Checkmate benchmark score: {checkmate_score}")
    # print(f"Checkmate (100-150) benchmark score: {checkmate_100_150_score}")
    # print(f"Checkmate full benchmark score: {(checkmate_score + checkmate_100_150_score) / 2}")
    # print(f"Stalemate benchmark score: {stalemate_score}")
    # print(f"Double check intention score: {dc_intention}")
    # print(f"Double check correctness score: {dc_score}")


if __name__ == "__main__":
    main(_parse_args(sys.argv[1:]))
