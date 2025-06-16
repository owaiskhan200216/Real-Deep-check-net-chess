import torch
import os
import numpy as np
from model import DeepCheckNet
from Self_Play_Loop import self_play_game, save_self_play_data
from Chess_Dataset_RL import ChessDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# === CONFIGURATION === #
n_games_per_iteration = 50
train_epochs = 10
self_play_data_file = "self_play_data.npz"
model_save_path = "DeepCheckNet_model.pth"
batch_size = 32
learning_rate = 1e-3
patience = 2

# === SETUP === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCheckNet().to(device)
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print("🔁 Loaded existing model.")
else:
    print("🚀 Starting with a new model.")

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
value_loss_fn = nn.MSELoss()

# === MAIN LOOP === #
iteration = 0
while True:
    iteration += 1
    print(f"\n============================\n🔁 Iteration {iteration}: Generating Self-Play\n============================")

    all_data = []
    for i in range(n_games_per_iteration):
        print(f"♟ Game {i + 1}/{n_games_per_iteration}")
        game_data = self_play_game(model)
        all_data.extend(game_data)

    save_self_play_data(all_data, self_play_data_file)
    print("✅ Self-play data saved.")

    print(f"\n============================\n🎯 Training Iteration {iteration}\n============================")
    


    dataset = ChessDataset(self_play_data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    best_loss = float("inf")
    patience_counter = 0

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.train()  # ✅ Ensure training mode is active before starting epochs

    for epoch in range(train_epochs):
        total_loss = 0.0

        for boards, policy_targets, value_targets in dataloader:
            model.train()  # ✅ Re-assert training mode every batch just in case

            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            policy_logits, value_preds = model(boards)

            policy_logits = policy_logits.view(policy_logits.size(0), -1)
            policy_targets = policy_targets.view(policy_targets.size(0), -1)
            policy_targets = policy_targets / policy_targets.sum(dim=1, keepdim=True).clamp(min=1e-6)

            policy_log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_targets * policy_log_probs).sum(dim=1).mean()
            value_loss = value_loss_fn(value_preds.view(-1), value_targets)
            loss = policy_loss + value_loss

            # print("policy_logits requires grad:", policy_logits.requires_grad)
            # print("value_preds requires grad:", value_preds.requires_grad)
            # print("loss requires grad:", loss.requires_grad)
            # print("loss grad_fn:", loss.grad_fn)
            # print("Any model param requires grad:", any(p.requires_grad for p in model.parameters()))


            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{train_epochs} | Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print("✅ Model improved and saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🚓 Early stopping this training iteration.")
                break
