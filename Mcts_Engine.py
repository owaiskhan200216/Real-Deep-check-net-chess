import math
import random
import numpy as np
import chess
import torch
from model import DeepCheckNet
from parse_pgn import board_to_tensor, move_to_index

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.is_expanded = False

    def expand(self, policy_probs):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            print(f"❌ No legal moves in position: {self.board.fen()}")

        for move in legal_moves:
            if move in self.children:
                continue
            try:
                idx = move_to_index(move, self.board)
                if idx is None:
                    print(f"⚠️ Skipping move (unencodable): {move}")
                    continue
                fy, fx, move_type = idx
                prior = policy_probs[move_type, fy, fx]
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = MCTSNode(new_board, parent=self, prior=prior)
            except Exception as e:
                print(f"❌ Error encoding move {move}: {e}")
                continue

        if not self.children:
            print("🚨 Node expansion failed — no valid children created.")
            print("Board FEN:", self.board.fen())
            print("Legal Moves:", [m.uci() for m in self.board.legal_moves])

        self.is_expanded = True

    def is_leaf(self):
        return not self.is_expanded

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, child, c_puct=1.5):
        prior_score = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        return child.value() + prior_score

    def select_child(self, c_puct=1.5):
        if not self.children:
            return None, None
        return max(self.children.items(), key=lambda item: self.ucb_score(item[1], c_puct))

    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)


def run_mcts(model, root, num_simulations=100, c_puct=1.5, add_dirichlet_noise=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if root.board.is_game_over():
        print(f"🌽 Root is a terminal position: {root.board.result()}")
        return

    if add_dirichlet_noise:
        alpha = 0.3
        epsilon = 0.25
        legal_moves = list(root.board.legal_moves)
        noise = np.random.dirichlet([alpha] * len(legal_moves))
        for i, move in enumerate(legal_moves):
            if move in root.children:
                root.children[move].prior = (
                    (1 - epsilon) * root.children[move].prior + epsilon * noise[i]
                )

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        while not node.is_leaf():
            move, next_node = node.select_child(c_puct)
            if next_node is None:
                print("⚠️ select_child returned no child; likely terminal node reached.")
                break
            node = next_node
            search_path.append(node)

        board_tensor = board_to_tensor(node.board)
        input_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)

        policy_logits, value = model(input_tensor)
        policy = torch.softmax(policy_logits[0], dim=0).detach().cpu().numpy()
        value = value.item()

        node.expand(policy)

        if not node.children:
            legal = list(node.board.legal_moves)
            if legal:
                fallback_move = random.choice(legal)
                print(f"⚠️ Fallback: pushing random move {fallback_move} in {node.board.fen()}")
                node.board.push(fallback_move)
                continue
            else:
                print("✅ Terminal node reached (checkmate or stalemate). No fallback move.")
                continue

        for node in reversed(search_path):
            node.backpropagate(value)


def select_move_by_visit_count(root, temperature=1.0):
    visits = [(move, child.visit_count) for move, child in root.children.items()]
    if not visits:
        return random.choice(list(root.board.legal_moves))

    moves, counts = zip(*visits)
    if temperature == 0:
        move = moves[np.argmax(counts)]
        probs = {m: 1.0 if m == move else 0.0 for m in moves}
        return move, probs

    adjusted_counts = [count ** (1 / temperature) for count in counts]
    total = sum(adjusted_counts)
    probs = [c / total for c in adjusted_counts]
    return random.choices(moves, weights=probs, k=1)[0], dict(zip(moves, probs))


def mcts_policy_tensor(root):
    probs_tensor = np.zeros((73, 8, 8), dtype=np.float32)
    for move, child in root.children.items():
        idx = move_to_index(move, root.board)
        if idx is None:
            continue
        fy, fx, move_type = idx
        probs_tensor[move_type, fy, fx] = child.visit_count
    probs_tensor /= probs_tensor.sum() + 1e-8
    return probs_tensor