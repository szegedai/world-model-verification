import random

from chess import Board

from utils import get_legal_moves


def random_move(board: Board, *args, **kwargs) -> str:
    return (random.choice(list(get_legal_moves(board))), None)

