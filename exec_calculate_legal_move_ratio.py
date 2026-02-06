import sys
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lightning import seed_everything
import chess

from models import get_model
from data_utils.chess_tokenizer import ChessTokenizer
from data_utils.data_collator import DataCollatorForLanguageModeling
from data_utils.line_dataset import LineByLineTextDataset
from moves.lm_move import lm_move
from utils import get_legal_moves



def _parse_args(args):
    parser = ArgumentParser("Chess LLM legal move ratio eval")

    # Model
    parser.add_argument("-m", "--model", type=str, default="chessgpt2_base", dest="model_type")
    parser.add_argument("-mp", "--model-path", type=str, required=True, dest="model_path")

    # Hyperparams
    parser.add_argument("-b", "--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--rap-prob", type=float, default=0.0, dest="rap_prob")
    parser.add_argument("-gpu", "--gpu-id", type=int, default=0, dest="gpu")
    parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")

    # Dataset access
    parser.add_argument("--vocab-file", type=str, dest="vocab_file")
    parser.add_argument("--data-file", type=str, dest="data_file")

    return parser.parse_args(args)

def main(conf):
    print(f"Legal move ratio eval for {conf.model_path}")
    seed_everything(conf.seed)

    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"

    model = get_model(conf.model_type, checkpoint_path=conf.model_path)
    model.eval()
    model.to(device)

    tokenizer = ChessTokenizer(vocab_file=conf.vocab_file)

    n_games = 1000

    lines = []
    with open(conf.data_file, "r") as f:
        lines = f.readlines()[:n_games]

    n_legal_moves = 0
    n_moves = 0

    # for line in tqdm(lines):
    for line in lines:
        moves = line.strip().split(" ")
        # n_moves_to_check = min(len(moves) - 5, 100)
        n_moves_to_check = len(moves) - 1
        game_prefix = ""
        tokenized_move_seq = [tokenizer.bos_token_id]
        board = chess.Board()
        for i, move in enumerate(moves[:n_moves_to_check]):
            game_prefix += move + " "
            tokenized_move_seq.extend(tokenizer.encode(move, False, False))
            board.push_uci(move)
            pred_move = lm_move(board, game_prefix, i+1, model, tokenizer, device)[0]

            if pred_move in get_legal_moves(board):
                n_legal_moves += 1
            n_moves += 1

    print(f"{conf.model_path}")
    print(f"Legal move ratio: {n_legal_moves / n_moves}")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(_parse_args(args))
