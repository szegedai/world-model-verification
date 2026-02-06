import sys
import json
from argparse import ArgumentParser
from functools import partial

import chess
from lightning import seed_everything
from tqdm import tqdm
from torch import nn

from data_utils.chess_tokenizer import ChessTokenizer
from models import get_model, setup_probe
from moves import MoveFunction
from moves.random_move import random_move
from moves.lm_move import lm_move
from moves.lm_move_topk import lm_move_topk
from moves.lm_move_topp import lm_move_topp
from moves.lm_move_corrected import lm_move_corrected
from moves.lm_move_adversarial import lm_move_adversarial
from moves.oracle_illegal_move import oracle_illegal_move
from moves.oracle_illegal_move_adaptive_topk import oracle_illegal_move_topk
from moves.oracle_probe_error import oracle_probe_error
from utils import get_legal_moves


def _parse_args(args):
    parser = ArgumentParser("Game simulation & evaluation")
    parser.add_argument("model_path", type=str)
    parser.add_argument("model_type", type=str)
    parser.add_argument("--vocab-file", type=str, dest="vocab_file")
    parser.add_argument("--n-games", type=int, default=1000, dest="n_games")
    parser.add_argument("--white-move-type",
                        type=str,
                        default="random_move",
                        choices=["random_move",
                                 "lm_move",
                                 "lm_move_corrected",
                                 "lm_move_adversarial",
                                 "oracle_illegal_move",
                                 "oracle_probe_error",
                                 "oracle_illegal_move_topk"],
                        dest="white_move_type")
    parser.add_argument("--black-move-type",
                        type=str,
                        default="lm_move",
                        choices=["lm_move", "lm_move_topk", "lm_move_topp"],
                        dest="black_move_type")
    parser.add_argument("--data-file", type=str, dest="data_file")
    parser.add_argument("--n-train-moves", type=int, default=0, dest="n_train_moves")
    parser.add_argument("--probe-path", type=str, dest="probe_path")
    parser.add_argument("--probe-layer-name", type=str, dest="probe_layer_name")
    parser.add_argument("--save-file", type=str, required=True, dest="save_file")
    parser.add_argument("--gpu", type=int, default=0, dest="gpu")
    parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")

    # Ablation
    parser.add_argument("--probe-error-target",
                        type=str,
                        choices=["cls_loss", "accuracy", "piece_accuracy"],
                        default="cls_loss",
                        dest="probe_error_target")

    # Sampling for black's move -- only active if black_move_type is a sampled
    # LM move
    parser.add_argument("--top-k-black", type=int, default=4, dest="top_k_black")
    parser.add_argument("--top-p-black", type=float, default=0.999, dest="top_p_black")

    return parser.parse_args(args)


def simulate_game(tokenizer: ChessTokenizer,
                  white_move_fn: MoveFunction,
                  black_move_fn: MoveFunction,
                  game_prefix: str = "",
                  min_single_moves: int = 0,
                  min_moves: int = 0
) -> None:
    board = chess.Board()

    # White moves first
    black_to_move = False
    move_counter = 1
    single_move_counter = 1
    move_sequence = [tokenizer.bos_token_id]
    last_move = ""
    white_supplementary = []

    # Prepare game prefix
    game_prefix = game_prefix.strip()
    if game_prefix != "":
        for move in game_prefix.split(" "):
            move_sequence.extend(tokenizer.encode(move, False, False))
            board.push_uci(move)

    # For non-empty initial game sequences
    if len(game_prefix) > 0: game_prefix += " "

    # Returned data
    result = None

    # Begin game
    while True:

        # Check is game is over
        if board.is_checkmate():
            if black_to_move:
                result = "white_wins"
            else:
                result = "black_wins"
            break

        if board.is_stalemate():
            result = "stalemate"
            break

        if board.is_insufficient_material():
            result = "insufficient_material"
            break

        if board.is_fivefold_repetition():
            result = "repetition"
            break

        if board.is_seventyfive_moves():
            result = "seventyfive_move"
            break

        # Make move
        if black_to_move:
            move, supp = black_move_fn(board, game_prefix, move_counter)
        else:
            move, supp = white_move_fn(board, game_prefix, move_counter)
            if supp: white_supplementary.append(supp)

        legal_moves = get_legal_moves(board)
        current_player = "black" if black_to_move else "white"
        last_move = move

        # Check the predicted move
        if move == "quit":
            result = f"{current_player}_quits"
            break
        if move not in legal_moves or move.startswith("error"):
            result = f"{current_player}_illegal_move"
            break

        # If the move is legal, push it and let the game continue
        board.push_uci(move)
        move_sequence.extend(tokenizer.encode(move, False, False))
        single_move_counter += 1
        if black_to_move: move_counter += 1
        black_to_move = not black_to_move
        game_prefix += move + " "
        # game_prefix += " " + move

    if move_counter <= min_moves or single_move_counter <= min_single_moves:
        return None
    else:
        return {
            "result": result,
            "n_moves": move_counter,
            "n_single_moves": single_move_counter,
            "last_move": last_move,
            "game_prefix": game_prefix,
            "white_supplementary": white_supplementary
        }


def run_simulation(conf):
    seed_everything(conf.seed)

    # Device, model, tokenizer
    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"
    model = get_model(conf.model_type, checkpoint_path=conf.model_path)
    model.eval()
    model.to(device)
    tokenizer = ChessTokenizer(conf.vocab_file)

    # LM plays as black by default - subject to change
    if conf.black_move_type == "lm_move":
        black_move_fn = partial(lm_move,
                                model=model,
                                tokenizer=tokenizer,
                                device=device)
    elif conf.black_move_type == "lm_move_topk":
        black_move_fn = partial(lm_move_topk,
                                model=model,
                                tokenizer=tokenizer,
                                device=device,
                                k=conf.top_k_black)
    elif conf.black_move_type == "lm_move_topp":
        black_move_fn = partial(lm_move_topp,
                                model=model,
                                tokenizer=tokenizer,
                                device=device,
                                p=conf.top_p_black)

    # White move function
    if conf.white_move_type == "random_move":
        white_move_fn = random_move
    elif conf.white_move_type == "lm_move":
        white_move_fn = partial(lm_move,
                                model=model,
                                tokenizer=tokenizer,
                                device=device)
    elif conf.white_move_type == "lm_move_corrected":
        white_move_fn = partial(lm_move_corrected,
                                model=model,
                                tokenizer=tokenizer,
                                device=device)
    elif conf.white_move_type == "lm_move_adversarial":
        white_move_fn = partial(lm_move_adversarial,
                                model=model,
                                tokenizer=tokenizer,
                                device=device)
    elif conf.white_move_type == "oracle_illegal_move":
        white_move_fn = partial(oracle_illegal_move,
                                model=model,
                                tokenizer=tokenizer,
                                device=device)
    elif conf.white_move_type == "oracle_illegal_move_topk":
        white_move_fn = partial(oracle_illegal_move_topk,
                                model=model,
                                tokenizer=tokenizer,
                                device=device,
                                top_k=4)
    elif conf.white_move_type == "oracle_probe_error":
        # Handle joint probe
        joint_probe = "probe" in conf.model_type
        if joint_probe:
            probe = nn.Identity()
        else:
            probe = setup_probe(model=model,
                                layer_name=conf.probe_layer_name,
                                device=device,
                                load_from_file=conf.probe_path)
        probe.to(device)
        probe.eval()
        white_move_fn = partial(oracle_probe_error,
                                model=model,
                                tokenizer=tokenizer,
                                layer_name=conf.probe_layer_name,
                                probe=probe,
                                device=device,
                                metric=conf.probe_error_target,
                                joint_probe=joint_probe)

    # Get ID game prefixes
    with open(conf.data_file, "r") as f: train_games = f.readlines()
    # Sorting needs to happen here for reproducibility - the ordering in
    # Python's set is nondeterministic due to nondeterministic hash values
    game_prefixes = sorted(list(set([" ".join(game.split(" ")[:2 * conf.n_train_moves])
                     for game in train_games if len(game.split(" ")) > 40])))

    # Simulate games
    logs = []
    for i in tqdm(range(conf.n_games)):
        logs.append(simulate_game(tokenizer=tokenizer,
                                  white_move_fn=white_move_fn,
                                  black_move_fn=black_move_fn,
                                  game_prefix=game_prefixes[i]))
        print(logs[-1])

    # Save logs
    with open(conf.save_file, "w+") as f:
        json.dump(logs, f)


if __name__ == "__main__":
    conf = _parse_args(sys.argv[1:])
    run_simulation(conf)


