import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block as used in AlphaZero
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class DeepCheckNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: 12 planes (6 pieces × 2 colors), 8x8 board
        self.input_conv = nn.Conv2d(12, 256, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(256)

        # 19 Residual Blocks (DeepMind used 19 for chess)
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(19)])

        # Policy head: 256 -> 2 -> 73 move types
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 8 * 8 * 73)  # 73 move types (including en passant)

        # Value head: 256 -> 1 -> FC -> scalar
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn_input(self.input_conv(x)))

        # Residual tower
        x = self.res_blocks(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = policy.view(-1, 73, 8, 8)  # Output shape: (batch, 73, 8, 8)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in range [-1, 1]

        return policy, value


# policy: shape (batch_size, 73, 8, 8)
# For every square on the board, predicts probability logits for each of the 73 move types (including en passant)
# value: shape (batch_size, 1)
# Scalar score representing expected outcome from current position
