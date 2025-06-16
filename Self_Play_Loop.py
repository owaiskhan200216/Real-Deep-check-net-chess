import chess
import random
import numpy as np
import torch

from model import DeepCheckNet
from parse_pgn import board_to_tensor, move_to_index
from Mcts_Engine import MCTSNode, run_mcts, select_move_by_visit_count, mcts_policy_tensor

# torch.set_grad_enabled(False)

def self_play_game(model, num_simulations=100, temperature_threshold=10):
    board = chess.Board()
    history = []
    move_number = 0

    while not board.is_game_over():
        root = MCTSNode(board.copy())
        run_mcts(model, root, num_simulations=num_simulations, add_dirichlet_noise=True)

        temperature = 1.0 if move_number < temperature_threshold else 0
        move, _ = select_move_by_visit_count(root, temperature=temperature)

        # Save state, policy (from MCTS), and placeholder value
        state_tensor = board_to_tensor(board)
        policy_tensor = mcts_policy_tensor(root)
        history.append((state_tensor, policy_tensor))

        board.push(move)
        move_number += 1

    # Final game result
    result = board.result()
    if result == '1-0':
        z = 1
    elif result == '0-1':
        z = -1
    else:
        z = 0

    # Assign values from perspective
    training_data = []
    perspective = 1
    for state_tensor, policy_tensor in history:
        training_data.append((state_tensor, policy_tensor, z * perspective))
        perspective *= -1

    return training_data

def save_self_play_data(data, filename):
    states, policies, values = zip(*data)
    states = np.stack(states)
    policies = np.stack(policies)
    values = np.array(values, dtype=np.float32)
    np.savez_compressed(filename, states=states, policies=policies, values=values)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepCheckNet().to(device)
    print("🧠 Starting self-play with randomly initialized model.")

    all_data = []
    num_games = 100
    for i in range(num_games):
        print(f"\n♟️ Game {i+1}/{num_games}")
        game_data = self_play_game(model)
        all_data.extend(game_data)

    save_self_play_data(all_data, "self_play_data.npz")
    print("✅ Self-play completed and saved to self_play_data.npz")