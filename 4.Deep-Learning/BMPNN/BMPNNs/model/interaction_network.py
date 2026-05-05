import torch.nn as nn
from .blocks import EdgeBlock, GlobalBlock
from .node_blocks import (
    BMPNodeBlock, ABMPNodeBlock, CBMPNodeBlock,
    BMP_SNNodeBlock, ABMP_SNNodeBlock, UMPNodeBlock
)

class InteractionNetwork(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, global_dim, dropout_rate, variant='BMP', task='Classification', num_classes=2):
        super().__init__()
        self.variant = variant
        self.task = task
        self.dropout_rate = dropout_rate
        self.edge_model = EdgeBlock(input_dim, edge_dim, hidden_dim, dropout_rate)
        self.global_model = GlobalBlock(hidden_dim, global_dim, dropout_rate)

        if variant == 'BMP':
            self.node_model = BMPNodeBlock(hidden_dim, dropout_rate)
        elif variant == 'ABMP':
            self.node_model = ABMPNodeBlock(input_dim, edge_dim, hidden_dim, dropout_rate)
        elif variant == 'CBMP':
            self.node_model = CBMPNodeBlock(hidden_dim, dropout_rate)
        elif variant == 'BMP+SN':
            self.node_model = BMP_SNNodeBlock(input_dim, hidden_dim, dropout_rate)
        elif variant == 'ABMP+SN':
            self.node_model = ABMP_SNNodeBlock(input_dim, edge_dim, hidden_dim, dropout_rate)
        elif variant == 'UMP':
            self.node_model = UMPNodeBlock(input_dim, hidden_dim, dropout_rate)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        if task == 'MultiTask':
            self.shared_multitask_mlp = self._build_prediction_head(hidden_dim, hidden_dim)
            self.class_head = self._build_prediction_head(hidden_dim, 1)
            self.active_head = self._build_prediction_head(hidden_dim, 1)
        else:
            self.output_head = nn.Linear(hidden_dim, 1)

    def _build_prediction_head(self, in_dim, out_dim):
        hidden_dim = max(in_dim // 2, 1)
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, u, batch, norm=None):
        src = x[edge_index[0]]
        dest = x[edge_index[1]]
        message = self.edge_model(src, dest, edge_attr)
        if self.variant == 'CBMP':
            from torch_geometric.utils import degree
            def compute_norm(edge_index, num_nodes):
                row, col = edge_index
                deg = degree(row, num_nodes=num_nodes)
                deg_inv_sqrt = deg.pow(-0.5) 
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                return norm
            norm = compute_norm(edge_index, x.size(0))
            x, x_weights = self.node_model(x, edge_index, message, norm)
        elif self.variant in ['ABMP', 'ABMP+SN']:
            x, x_weights = self.node_model(x, edge_index, edge_attr, message)
        else: 
            x, x_weights = self.node_model(x, edge_index, message)

        graph_embedding = self.global_model(x, u, batch)
        if self.task == 'MultiTask':
            shared_embedding = self.shared_multitask_mlp(graph_embedding)
            outputs = {
                'class_logits': self.class_head(shared_embedding),
                'active_pred': self.active_head(shared_embedding),
                'graph_embedding': graph_embedding,
                'shared_embedding': shared_embedding,
            }
            return outputs, x_weights
        out = self.output_head(graph_embedding)
        return out, x_weights
