from typing import Callable, Tuple, Any

from chess import Board


# Type alias for move function: input is a chess board, the game prefix and the
# move number, output is a (valid or invalid) move
MoveFunction = Callable[[Board, str, int], Tuple[str, Any]]
