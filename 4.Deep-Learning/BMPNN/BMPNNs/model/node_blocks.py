import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.utils import softmax

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.W_src = nn.Linear(input_dim, hidden_dim, bias=False)  
        self.W_dest = nn.Linear(input_dim, hidden_dim, bias=False) 
        self.W_edge = nn.Linear(edge_dim, hidden_dim, bias=False) 
        self.attn_vector = nn.Parameter(torch.Tensor(1, hidden_dim)) 
        nn.init.xavier_uniform_(self.attn_vector)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, src, dest, edge_attr, edge_index):   
        src_transformed = self.W_src(src) 
        dest_transformed = self.W_dest(dest) 
        edge_transformed = self.W_edge(edge_attr)  
        edge_scores = self.leaky_relu(
            torch.matmul(src_transformed + dest_transformed + edge_transformed, self.attn_vector.t())
        ) 
        _, col = edge_index 
        attn_weights = softmax(edge_scores, col)
        return attn_weights
class BMPNodeBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.attention_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, x, edge_index, message):
        row, col = edge_index
        forward = scatter(message, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(message, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([forward, backward], dim=1)
        out = self.node_mlp(out)
        return out, self.attention_mlp(out).view(-1)
class UMPNodeBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),  
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
                   
        )
        self.attention_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, x, edge_index, message):
        row, col = edge_index
        out = x[row]  
        out = torch.cat([message, out], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp_2(out)
        return out, self.attention_mlp(out).view(-1)
class ABMPNodeBlock(nn.Module): 
    def __init__(self, input_dim, edge_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.attention = AttentionMechanism(input_dim, edge_dim, hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, message):
        row, col = edge_index
        src, dest = x[row], x[col]
        attn_weights = self.attention(src, dest, edge_attr, edge_index)
        message = message * attn_weights.view(-1, 1)
        forward = scatter(message, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(message, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([forward, backward], dim=1)
        out = self.node_mlp(out)
        return out, self.attention_mlp(out).view(-1)


class CBMPNodeBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr, norm):
        row, col = edge_index
        edge_attr = norm.unsqueeze(1) * edge_attr
        forward = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(edge_attr, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([forward, backward], dim=1)
        out = self.node_mlp(out)
        return out, self.attention_mlp(out).view(-1)

class BMP_SNNodeBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim + 2 * hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()  
        )

    def forward(self, x, edge_index, message):
        row, col = edge_index
        forward = scatter(message, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(message, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([x, forward, backward], dim=1)
        out = self.node_mlp(out)
        return out, self.attention_mlp(out).view(-1) 
class ABMP_SNNodeBlock(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.attention = AttentionMechanism(input_dim, edge_dim, hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + input_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()  
        )
        self.attention = AttentionMechanism(input_dim, edge_dim, hidden_dim)  
    def forward(self, x, edge_index, edge_attr, message):
        row, col = edge_index
        src, dest = x[row], x[col]
        attn_weights = self.attention(src, dest, edge_attr, edge_index)
        message = message * attn_weights.view(-1, 1)
        forward = scatter(message, col, dim=0, dim_size=x.size(0), reduce='max')
        backward = scatter(message, row, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([x, forward, backward], dim=1)
        out = self.node_mlp(out)
        return out, self.attention_mlp(out).view(-1) 
