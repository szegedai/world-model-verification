import os
from typing import List

import chess
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.utils.data.dataset import Dataset

from data_utils.chess_tokenizer import ChessTokenizer
from utils import get_legal_moves


logger = logging.getLogger(__name__)


def get_rap_data(transcript_line: str) -> List[str]:
    board = chess.Board()
    moves = transcript_line.split(" ")
    rap_list = []

    for move in moves:
        # There was some unknown error with benchmarks that required this check
        # and I was in a haste, so instead of figuring out the root cause I
        # just patched it... it works though
        if move == "": continue

        starting_square = move[:2]
        piece = board.piece_at(chess.parse_square(starting_square)).symbol() # type: ignore
        rap_list.append(piece.upper())
        board.push_uci(move)

    return rap_list


def get_valid_token_probability_distributions(tokenized_sequence: List[int],
                                              move_end_indices: List[int],
                                              tokenizer: ChessTokenizer,
                                              annotated: bool
) -> List[List[int]]:
    """TODO"""
    assert len(tokenized_sequence) == len(move_end_indices)

    board = chess.Board()
    last_move = ""
    annotation_done = False
    annotation = ""
    valid_tokens = []

    for i in range(len(tokenized_sequence)):

        move_part = tokenizer.decode_token(tokenized_sequence[i])

        if move_part in ['P', 'B', 'N', 'R', 'Q', 'K']:
            annotation = move_part
        else:
            last_move += move_part

        if move_end_indices[i] == 1:
            if last_move != tokenizer.bos_token: board.push_uci(last_move)
            last_move = ""
            annotation_done = False

        legal_moves = get_legal_moves(board)

        if annotated:
            # PAD after EOS or PAD
            if move_part == tokenizer.eos_token or move_part == tokenizer.pad_token:
                possible_tokens = [tokenizer.pad_token]

            # Game end
            elif len(legal_moves) == 0:
                possible_tokens = [tokenizer.eos_token]

            # Annotation
            elif not annotation_done:
                possible_tokens = []
                for move in legal_moves:
                    possible_tokens.append(board.piece_at(chess.parse_square(move[:2])).symbol().upper()) # type: ignore
                annotation_done = True

            # Move start
            elif annotation_done and len(last_move) == 0:
                possible_tokens = [move[:2] for move in legal_moves
                                   if board.piece_at(chess.parse_square(move[:2])).symbol().upper() == annotation] # type: ignore

            # Move end
            elif annotation_done and len(last_move) == 2:
                possible_tokens = [move[2:4] for move in legal_moves if move.startswith(last_move)]

            # Promotion
            elif annotation_done and len(last_move) == 4 and last_move != tokenizer.eos_token:
                possible_tokens = ['q', 'r', 'b', 'n']

        else:

            # PAD after EOS or PAD
            if move_part == tokenizer.eos_token or move_part == tokenizer.pad_token:
                possible_tokens = [tokenizer.pad_token]

            # Game end
            elif len(legal_moves) == 0:
                possible_tokens = [tokenizer.eos_token]

            # Move start
            elif last_move == "":
                possible_tokens = [move[:2] for move in legal_moves]

            # Move end
            elif len(last_move) == 2:
                possible_tokens = [move[2:4] for move in legal_moves if move.startswith(last_move)]

            # Promotion
            elif len(last_move) == 4 and last_move != tokenizer.eos_token:
                possible_tokens = ['q', 'r', 'b', 'n']

        valid_tokens.append(set(possible_tokens))

    valid_token_probs = []
    for token_set in valid_tokens:
        prob_distribution = [0] * len(tokenizer.vocab)
        prob = 1 / len(token_set)

        for token in token_set:
            prob_distribution[tokenizer.vocab[token]] = prob

        valid_token_probs.append(prob_distribution)

    # return valid_tokens # For debugging
    return valid_token_probs

class LineByLineTextDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 file_path,
                 block_size,
                 max_instances=None,
                 rap_prob=0.0,
                 valid_token_distribution: bool = False,
                 board_state: bool = False):
        assert os.path.isfile(file_path)
        self.rap_prob = rap_prob
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_len = 0
        self.valid_token_distribution = valid_token_distribution
        self.board_state = board_state

        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            if max_instances:
                self.lines = self.lines[:int(max_instances)]

        # print("Calculating max. game length...")
        # for line in tqdm(self.lines):
        #     encoded_line, _ = self.tokenizer.encode(line, get_move_end_positions=True)
        #     self.max_len = max(self.max_len, len(encoded_line))
        # print(f"Max. game length: {self.max_len}\n")

        # if self.rap_prob:
        #     rap_dir = path.join(path.dirname(path.dirname(file_path)), "rap")
        #     rap_file = path.join(rap_dir, path.splitext(path.basename(file_path))[0] + ".npy")
        #     self.rap_data = np.load(rap_file, allow_pickle=True)[:max_instances]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        output_dict = {}
        encoding = self.tokenizer([self.lines[i]], add_special_tokens=True, truncation=True, max_len=self.block_size)
        example = encoding["input_ids"][0]
        end_position = encoding["end_positions"][0]

        if self.rap_prob:
            rap_data = get_rap_data(self.lines[i])
            example, end_position, piece_type_posns = self._process_rap(rap_data, example, end_position)
            output_dict['piece_type_posns'] = piece_type_posns

        if self.valid_token_distribution:
            valid_token_distributions = get_valid_token_probability_distributions(example, end_position, self.tokenizer, annotated=(self.rap_prob > 0))
            output_dict['valid_token_distributions'] = torch.tensor(valid_token_distributions)

        # Padding
        # example.extend([self.tokenizer.pad_token_id] * (self.max_len - len(example)))
        # end_position.extend([self.tokenizer.pad_token_id] * (self.max_len - len(end_position)))

        # Board state
        if self.board_state:
            from utils.probing import convert_moves_to_fen, convert_fen_to_labels
            data_dict = {
                'input_ids': example,
                'separator_ind': end_position
            }
            board_state_fens = convert_moves_to_fen(data_dict, self.tokenizer)
            board_states = [convert_fen_to_labels(fen)
                            for fen in board_state_fens]
            output_dict['board_states'] = torch.stack(board_states)

        output_dict['input_ids'] = torch.tensor(example)
        output_dict['separator_ind'] = end_position
        return output_dict

    def _process_rap(self, rap_list, example, end_positions):
        use_piece_type_list = np.random.choice([0, 1], size=(len(rap_list),), p=[1 - self.rap_prob, self.rap_prob])
        piece_type_list = [self.tokenizer.vocab[piece_type] if (piece_type is not None and use_piece_type) else -1
                           for (piece_type, use_piece_type) in zip(rap_list, use_piece_type_list)]

        assert (len(piece_type_list) == sum(end_positions) - 1)

        mod_example = list(example)
        mod_end_positions = list(end_positions)

        offset = 1
        move_counter = 0
        piece_type_posns = []
        for idx, end_position in enumerate(end_positions):
            # Check if it's end position
            if (end_position == 1) and (move_counter < len(piece_type_list)):
                piece_type = piece_type_list[move_counter]
                if piece_type != -1:
                    # Insert piece type
                    mod_example = mod_example[:idx + offset] + [piece_type] + mod_example[idx + offset:]
                    mod_end_positions = mod_end_positions[:idx + offset] + [0] + mod_end_positions[idx + offset:]
                    piece_type_posns.append(idx + offset)

                    offset += 1

                move_counter += 1

        return mod_example, mod_end_positions, piece_type_posns

    def get_last_mention_idx(self, example):
        tokenizer = self.tokenizer
        tokens = [tokenizer.id2symbol[idx] for idx in example]
        is_posn_list = [1 if len(token) == 2 else 0 for token in tokens]
        is_move_end_list = [0] * len(is_posn_list)
        counter = 0
        for idx, posn_token in enumerate(is_posn_list):
            if posn_token:
                counter += 1
                if counter % 2 == 0:
                    is_move_end_list[idx] = 1

        last_mention_dict = {}
        last_mention_list = []
        for idx, (token, is_posn, is_move_end) in enumerate(zip(tokens, is_posn_list, is_move_end_list)):
            last_mention_idx = -1
            if is_posn:
                if is_move_end:
                    last_mention_dict[token] = idx
                else:
                    # Starting token
                    if token in last_mention_dict:
                        last_mention_idx = last_mention_dict[token]

            last_mention_list.append(last_mention_idx)

        return last_mention_list

