import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool

class EdgeBlock(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2 + edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, src, dest, edge_attr):
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)

class GlobalBlock(nn.Module):
    def __init__(self, hidden_dim, global_dim, dropout_rate):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, u, batch):
        u_x = global_max_pool(x, batch)
        return self.global_mlp(torch.cat([u_x, u], dim=1))
