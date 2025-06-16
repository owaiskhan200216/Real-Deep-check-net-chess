# chess_engine.py
import torch
from model import DeepCheckNet
from parse_pgn import board_to_tensor, move_to_index
import chess
import numpy as np

# === Load Model === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCheckNet().to(device)
model.load_state_dict(torch.load("DeepCheckNet_model.pth", map_location=device))
model.eval()

# === Game Loop === #
board = chess.Board()

while not board.is_game_over():
    print(board)
    print()

    if board.turn == chess.WHITE:
        move_uci = input("Your move: ")
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                print("Illegal move. Try again.")
                continue
        except:
            print("Invalid input. Try again.")
            continue
    else:
        # === Model's Turn === #
        board_tensor = board_to_tensor(board)
        input_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, _ = model(input_tensor)

        # Output shape: (1, 73, 8, 8) → squeeze to (73, 8, 8), then permute to (8, 8, 73)
        policy = policy_logits.squeeze(0).permute(1, 2, 0).cpu().numpy()

        move_scores = []
        for move in board.legal_moves:
            idx = move_to_index(move, board)
            if idx is not None:
                fy, fx, move_type = idx
                score = policy[fy, fx, move_type]
                move_scores.append((score, move))

        if move_scores:
            move_scores.sort(reverse=True)
            best_move = move_scores[0][1]
            print(f"Model plays: {best_move.uci()}")
            board.push(best_move)
        else:
            print("No legal move found by model.")
            break

print(board)
print("Game over!")
print(board.result())