import torch
from torch.utils.data import Dataset

from growing_transformer import Growing
from growing_transformer.mlp import GrowingMLP as MLP


class SimpleModel(Growing):
    def __init__(self, n_in, n_out, h, a, b, config={}):
        super().__init__(config)
        self.a = MLP(n_in, h, a, config=config)
        self.b = MLP(h, n_out, b, config=config)
        self.dropout = torch.nn.Dropout()

    def forward(self, x):
        x = self.a(x)
        x = self.dropout(x)
        x = self.b(x)
        return x


class SineToyDataset(Dataset):
    def __init__(self, n_samples, amp=0.8, freq=1.0, phase=0.0):
        self.x = torch.empty(n_samples, 1)
        noise_y = torch.empty(n_samples, 1)
        torch.nn.init.uniform_(self.x, -5, 5)
        torch.nn.init.normal_(noise_y, 0, 0.1)
        self.y = torch.sin(phase + self.x * freq) * amp + noise_y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
