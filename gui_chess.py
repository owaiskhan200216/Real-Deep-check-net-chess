import pygame
import chess
import os
import torch
import numpy as np
import random

from parse_pgn import board_to_tensor, move_to_index
from model import DeepCheckNet

# Config
WIDTH, HEIGHT = 512, 640  # Extra height for move display
SQUARE_SIZE = 64
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
BLACK = (0, 0, 0)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCheckNet().to(device)
model.load_state_dict(torch.load("DeepCheckNet_model.pth", map_location=device))
model.eval()

# Load piece images
PIECES = {}
def load_images():
    pieces = ["P", "N", "B", "R", "Q", "K"]
    colors = ["w", "b"]
    for color in colors:
        for piece in pieces:
            name = f"{color}{piece}"
            path = os.path.join("images", f"{name}.png")
            image = pygame.image.load(path)
            PIECES[name] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(screen, board):
    colors = [WHITE, BROWN]
    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(file*SQUARE_SIZE, rank*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            img = PIECES[f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"]
            screen.blit(img, pygame.Rect(col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_move_log(screen, move_log, font):
    pygame.draw.rect(screen, WHITE, pygame.Rect(0, 512, 512, 128))
    lines = []
    for i in range(0, len(move_log), 2):
        line = f"{(i // 2) + 1}. {move_log[i]}"
        if i + 1 < len(move_log):
            line += f" {move_log[i+1]}"
        lines.append(line)

    for i, line in enumerate(lines[-4:]):
        text_surface = font.render(line, True, BLACK)
        screen.blit(text_surface, (10, 518 + i * 24))

def pixel_to_square(x, y):
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    return chess.square(file, rank)

def get_model_move(board):
    board_tensor = board_to_tensor(board)
    input_tensor = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, _ = model(input_tensor)

    policy = policy_logits.squeeze(0).permute(1, 2, 0).cpu().numpy()

    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = -float('inf')

    for move in legal_moves:
        idx = move_to_index(move, board)
        if idx is None:
            continue
        fy, fx, move_type = idx
        if move_type >= policy.shape[2]:
            continue
        score = policy[fy, fx, move_type]
        if score > best_score:
            best_score = score
            best_move = move

    return best_move if best_move else random.choice(legal_moves)

def get_promotion_choice(screen, font):
    prompt = "Promote to: Q (Queen) or N (Knight)"
    text_surface = font.render(prompt, True, BLACK)
    pygame.draw.rect(screen, WHITE, pygame.Rect(0, HEIGHT // 2 - 30, WIDTH, 60))
    screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, HEIGHT // 2 - 10))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return chess.QUEEN
                elif event.key == pygame.K_n:
                    return chess.KNIGHT
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DeepCheckNet GUI")
    load_images()
    font = pygame.font.SysFont(None, 24)

    board = chess.Board()
    selected_square = None
    clock = pygame.time.Clock()
    running = True

    mode = None
    while mode not in ['1', '2']:
        print("Select mode:")
        print("1: Human vs Model")
        print("2: Model vs Model")
        mode = input("Enter 1 or 2: ")

    is_human_white = (mode == '1')
    move_log = []

    while running:
        clock.tick(30)
        draw_board(screen, board)
        draw_move_log(screen, move_log, font)
        pygame.display.flip()

        if board.is_game_over():
            print("Game Over!", board.result())
            with open("game_log.txt", "w") as f:
                f.write("[Event \"Model Game\"]\n")
                f.write(f"[Result \"{board.result()}\"]\n\n")
                for i, move in enumerate(move_log):
                    if i % 2 == 0:
                        f.write(f"{(i // 2) + 1}. ")
                    f.write(f"{move} ")
                f.write("\n")
            pygame.time.wait(2000)
            running = False
            continue

        if is_human_white and board.turn == chess.WHITE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    if y >= 512:
                        continue
                    square = pixel_to_square(x, y)
                    if selected_square is None:
                        if board.piece_at(square) and board.color_at(square) == chess.WHITE:
                            selected_square = square
                    else:
                        move = chess.Move(selected_square, square)
                        if move in board.legal_moves:
                            board.push(move)
                            move_log.append(move.uci())
                            selected_square = None
                        else:
                            if chess.square_rank(square) == 7:
                                promo_piece = get_promotion_choice(screen, font)
                                move = chess.Move(selected_square, square, promotion=promo_piece)
                                if move in board.legal_moves:
                                    board.push(move)
                                    move_log.append(move.uci())
                            selected_square = None
        else:
            pygame.time.wait(300)
            move = get_model_move(board)
            board.push(move)
            move_log.append(move.uci())

    pygame.quit()

if __name__ == "__main__":
    main()
