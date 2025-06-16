import torch
from torch.utils.data import Dataset
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.states = torch.tensor(data['states'], dtype=torch.float32)
        self.policies = torch.tensor(data['policies'], dtype=torch.float32)
        self.values = torch.tensor(data['values'], dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]
