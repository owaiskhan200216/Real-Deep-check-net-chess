import chess
import torch
import numpy as np

# AlphaZero-style move encoding constants

SLIDING_DIRECTIONS = [
    (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
    (-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0),
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (0, -1), (0, -2), (0, -3), (0, -4), (0, -5), (0, -6), (0, -7),
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
    (-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5), (-6, 6), (-7, 7),
    (1, -1), (2, -2), (3, -3), (4, -4), (5, -5), (6, -6), (7, -7),
    (-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5), (-6, -6), (-7, -7),
]

KNIGHT_OFFSETS = [
    (1, 2), (2, 1), (-1, 2), (-2, 1),
    (1, -2), (2, -1), (-1, -2), (-2, -1)
]

PROMOTION_OFFSETS = [(0, 1), (-1, 1), (1, 1)]
PROMOTION_PIECES = [chess.KNIGHT, chess.QUEEN]
NUM_MOVE_TYPES = 73

def square_to_coords(square):
    return chess.square_file(square), chess.square_rank(square)

def coords_to_square(file, rank):
    if 0 <= file < 8 and 0 <= rank < 8:
        return chess.square(file, rank)
    return None

def move_to_index(move, board=None):
    from_sq = move.from_square
    to_sq = move.to_square
    fx, fy = square_to_coords(from_sq)
    tx, ty = square_to_coords(to_sq)
    dx, dy = tx - fx, ty - fy

    for i, (sx, sy) in enumerate(SLIDING_DIRECTIONS):
        if dx == sx and dy == sy:
            return fy, fx, i

    for i, (kx, ky) in enumerate(KNIGHT_OFFSETS):
        if dx == kx and dy == ky:
            return fy, fx, 56 + i

    if move.promotion in PROMOTION_PIECES:
        for i, (px, py) in enumerate(PROMOTION_OFFSETS):
            if dx == px and dy == py:
                piece_index = PROMOTION_PIECES.index(move.promotion)
                return fy, fx, 64 + (i * 2) + piece_index

    # Castling
    if move == chess.Move.from_uci("e1g1") or move == chess.Move.from_uci("e8g8"):
        return fy, fx, 70
    if move == chess.Move.from_uci("e1c1") or move == chess.Move.from_uci("e8c8"):
        return fy, fx, 71

    if board and board.is_en_passant(move):
        return fy, fx, 72

    return None

def board_to_tensor(board):
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        piece_type = piece.piece_type - 1
        color_offset = 0 if piece.color == chess.WHITE else 6
        row = 7 - (square // 8)
        col = square % 8
        planes[color_offset + piece_type, row, col] = 1.0

    return planes

def move_to_policy_tensor(move, board):
    idx = move_to_index(move, board)
    if idx is None:
        return None
    fy, fx, move_type = idx
    policy_tensor = np.zeros((8, 8, 73), dtype=np.float32)
    policy_tensor[fy, fx, move_type] = 1.0
    return policy_tensor

def parse_pgn_file(pgn_path, max_games=2000):
    with open(pgn_path, 'r') as f:
        game_counter = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None or game_counter >= max_games:
                break

            board = game.board()
            result = game.headers.get("Result")
            if result == "1-0":
                value = 1
            elif result == "0-1":
                value = -1
            else:
                value = 0

            for move in game.mainline_moves():
                policy_tensor = move_to_policy_tensor(move, board)
                if policy_tensor is not None:
                    board_tensor = board_to_tensor(board)
                    yield board_tensor, policy_tensor, value
                board.push(move)

            game_counter += 1

