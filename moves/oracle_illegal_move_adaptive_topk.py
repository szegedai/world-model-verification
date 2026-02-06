import math
from typing import List, Union, Dict
from copy import deepcopy

import chess
import torch
from chess import Board

from data_utils.chess_tokenizer import ChessTokenizer
from models import ChessLM
from utils import get_legal_moves, tokenize_sequence
from moves.random_move import random_move
from moves.lm_move import chess_lm_move


def random_move_smart(board: chess.Board,
                      model,
                      game_sequence,
                      tokenizer,
                      device
) -> str:
    """Returns a random legal move that keeps the game going for at least 1 more move."""
    for legal_move in get_legal_moves(board):
        board_new = deepcopy(board)
        board_new.push_uci(legal_move)
        if not board_new.is_game_over():
            new_sequence = [
                *tokenize_sequence(game_sequence, tokenizer),
                *tokenizer.encode(legal_move, False, False)
            ]
            lm_move = chess_lm_move(model, tokenizer, new_sequence, device)[0]
            if lm_move not in get_legal_moves(board_new):
                return legal_move
            board_new2 = deepcopy(board_new)
            board_new2.push_uci(lm_move)
            if not board_new2.is_game_over():
                return legal_move


def get_all_moves_with_probs(tokenized_sequence: List[int],
                             model: ChessLM,
                             tokenizer: ChessTokenizer,
                             device: torch.device,
                             eps: float = 1e-8,
                             batch_size: int = 128,
                             check_promotion: bool = True,
                             top_k: int = 4,
) -> Dict[str, float]:

    # Special tokens
    special_tokens = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id
    ]

    # Promotion tokens
    promotion_tokens = [
        tokenizer.encode_token('q'),
        tokenizer.encode_token('r'),
        tokenizer.encode_token('b'),
        tokenizer.encode_token('n')
    ]

    # Prediction part 1: starting square
    token_probs_p1 = {}

    # This part includes a hotfix: with transformers==4.46.3, GPT2LMHeadModel
    # could handle single (non-batched) sequences, but in transformers==4.55.3,
    # this has been changed, so only batched sequences can be passed to the
    # model. I love the absolute pile of dogshit that is transformers :)
    sequence = torch.tensor([tokenized_sequence]).to(device)
    out = model(sequence)
    if len(out.shape) > 2:
        out = out[0]

    out_probs = out[-1, :]
    topk_logits, topk_indices = torch.topk(out_probs, k=top_k, dim=-1)
    topk_probs = torch.softmax(topk_logits, dim=-1)
    # out_probs_sorted = torch.argsort(out_probs, descending=True)[:top_k]
    # out_probs_sorted = torch.softmax(out_probs_sorted, dim=-1)
    for out_token, prob in zip(topk_indices, topk_probs):
        # prob = out_probs[out_token]
        if prob > eps:
            # move_prefix = tokenizer.decode_token(out_token)
            token_probs_p1[out_token.item()] = prob.item()

        # Because probs are sorted, we can break the first time a prob is
        # lower than eps
        else: break

    # Prediction part 2: ending square
    token_probs_p2 = {}
    p2_sequences = []
    p1_tokens = []
    for token in token_probs_p1.keys():
        if token not in special_tokens:
            p2_sequences.append([*tokenized_sequence, token])
            p1_tokens.append(token)

    # For now, let's not do batched prediction in part 2 - the batch size here
    # will be <= 74, hopefully it will be fine :)
    p2_sequences = torch.tensor(p2_sequences).to(device)
    out = model(p2_sequences)
    out_probs = torch.softmax(out[:, -1, :], 1)

    for i in range(len(out_probs)):
        p1_token = p1_tokens[i]
        base_prob = token_probs_p1[p1_token]
        # out_prob = out_probs[i] # horrible variable name
        topk_logits, topk_indices = torch.topk(out_probs[i], k=top_k, dim=-1)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        # out_probs_sorted = torch.argsort(out_prob, descending=True)

        for topk_prob, topk_token in zip(topk_probs, topk_indices):
            prob = topk_prob.item() * base_prob
            if prob > eps:
                token_probs_p2[(p1_token, topk_token.item())] = prob
            else: break


    # Only check promotion if specified
    if check_promotion:

        # Hotfix for now...
        if len(token_probs_p2) == 0:
            return {}

        # Prediction part 3: promotion
        token_probs_p3 = {}
        p3_sequences = []
        p2_token_pairs = []
        for (start, end) in token_probs_p2.keys():
            if end not in special_tokens:
                p3_sequences.append([*tokenized_sequence, start, end])
                p2_token_pairs.append((start, end))

        # There are *a lot* of possibilities here, so the LM prediction is done
        # in batches
        p3_sequences = torch.tensor(p3_sequences).to(device)
        out = []
        for i in range(math.ceil(p3_sequences.shape[0] / batch_size)):
            batch = p3_sequences[i * batch_size : (i + 1) * batch_size, :]
            out.extend(model(batch).cpu().detach())
        out = torch.stack(out)
        out_argmax = torch.argmax(out[:, -1, :], 1)

        # Only consider promotion as part of the move if it would be the top-1
        # prediction by the LM. This is extremely rare.
        for i in range(len(out_argmax)):
            p2_token_pair = p2_token_pairs[i]
            base_prob = token_probs_p2[p2_token_pair]
            out_token = out_argmax[i].item()

            if out_token in promotion_tokens:
                token_probs_p3[(*p2_token_pair, out_token)] = prob

                # Delete move prefix without promotion
                if p2_token_pair in token_probs_p2:
                    del token_probs_p2[p2_token_pair]

    all_moves = {}

    # Part 1
    for token, prob in token_probs_p1.items():
        if token in special_tokens:
            move = tokenizer.decode_token(token)
            all_moves[move] = prob

    # Part 2
    for (start, end), prob in token_probs_p2.items():
        if start not in special_tokens:
            move = "".join([tokenizer.decode_token(start),
                            tokenizer.decode_token(end)])
            all_moves[move] = prob

    # Part 3
    if check_promotion:
        for (start, end, prom), prob in token_probs_p3.items():
            if start not in special_tokens and prom in promotion_tokens:
                move = "".join([tokenizer.decode_token(start),
                                tokenizer.decode_token(end),
                                tokenizer.decode_token(prom)])

                all_moves[move] = prob

                if move[:-1] in all_moves.keys():
                    del all_moves[move[:-1]]

    return all_moves


def search_for_illegal_move(game_sequence: List[str],
                            model: ChessLM,
                            tokenizer: ChessTokenizer,
                            device: torch.device,
                            eps: float,
                            check_promotion: bool = True,
                            top_k: int = 4
) -> str:
    # Setup board
    board = chess.Board()
    for move in game_sequence:
        board.push_uci(move)

    potential_moves = []
    top_move = None
    top_illegal_move_prob = 0

    # Only consider legal moves that don't end the game - the goal is to force
    # an illegal move, not to end the game
    for move in get_legal_moves(board):
        board_copy = deepcopy(board)
        board_copy.push_uci(move)
        if board_copy.is_game_over(): continue

        # Only consider moves to which the top LM response also does not
        # end the game
        extended_sequence = [
            *tokenize_sequence(game_sequence, tokenizer),
            *tokenizer.encode(move, False, False)
        ]
        lm_move = chess_lm_move(model, tokenizer, extended_sequence, device)[0]

        new_legal_moves = get_legal_moves(board_copy)

        # If the top LM move is illegal, the attack is complete, the
        # optimization part would be redundant
        # if lm_move not in new_legal_moves:
        # ^not true here, we attack sampling models
        if False:
            return move, 1
        else:
            # board_copy2 = deepcopy(board_copy)
            # board_copy2.push_uci(lm_move)

            # # Discard the move if the top response ends the game
            # if board_copy2.is_game_over(): continue

            new_game_sequence = [*game_sequence, move]
            tokenized_sequence = tokenize_sequence(new_game_sequence,
                                                   tokenizer)
            move_probs = get_all_moves_with_probs(tokenized_sequence,
                                                  model,
                                                  tokenizer,
                                                  device,
                                                  eps,
                                                  batch_size=128,
                                                  check_promotion=check_promotion,
                                                  top_k=top_k)
            sum_illegal_move_prob = 0
            for new_move, prob in move_probs.items():
                if new_move not in new_legal_moves:
                    sum_illegal_move_prob += prob
            sum_illegal_move_prob = sum_illegal_move_prob
            if sum_illegal_move_prob > top_illegal_move_prob:
                top_move = move
                top_illegal_move_prob = sum_illegal_move_prob


    if top_move is None:
        # top_move = random_move(board)[0]
        print("encountered a None top move")
        top_move = random_move_smart(board, model, game_sequence, tokenizer, device)
        if top_move is None:
            print("there is no move that keeps the game going")
            top_move = random_move(board)[0]

    return top_move, top_illegal_move_prob


def oracle_illegal_move_topk(board: Board,
                             game_prefix: str,
                             move_number: int,
                             model: ChessLM,
                             tokenizer: ChessTokenizer,
                             device: Union[torch.device, str],
                             eps: float = 1e-4,
                             top_k: int = 4
) -> str:
    """Oracle (illegal move) adapter so that it conforms to `moves.MoveFunction`."""
    game_prefix_filtered = [s for s in game_prefix.split(" ") if len(s) > 0]
    return search_for_illegal_move(game_prefix_filtered,
                                   model,
                                   tokenizer,
                                   device,
                                   eps,
                                   check_promotion=True,
                                   top_k=top_k)

