import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    timesteps = timesteps.reshape(-1).float()
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps[:, None] * frequencies[None]
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


class NodeWiseLSTMBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        h_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.h_dim = int(h_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.lstm_cell = nn.LSTMCell(input_size=input_dim, hidden_size=h_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, feature_dim = values.shape
        values = self.norm(values).reshape(batch_size * num_nodes, feature_dim)
        hidden = hidden.reshape(batch_size * num_nodes, self.h_dim)
        cell = cell.reshape(batch_size * num_nodes, self.h_dim)
        hidden, cell = self.lstm_cell(values, (hidden, cell))
        hidden = self.dropout(hidden)
        return (
            hidden.reshape(batch_size, num_nodes, self.h_dim),
            cell.reshape(batch_size, num_nodes, self.h_dim),
        )

    def init_states(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, num_nodes, self.h_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )


class GraphNodeMixingBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout=0.1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, node_features, adjacency):
        adjacency = adjacency.clamp_min(0)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        messages = torch.bmm(adjacency, node_features)
        return self.norm(node_features + self.dropout(self.msg_mlp(messages)))


class SpatialBlock(nn.Module):
    def __init__(self, u_dim, h_dim, spatial_dim, msg_dim, dropout=0.1, eps=1e-6):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(u_dim + h_dim, spatial_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim, spatial_dim),
        )
        self.graph_mix = GraphNodeMixingBlock(spatial_dim, msg_dim, dropout, eps)
        self.out_proj = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim, spatial_dim),
        )
        self.norm = nn.LayerNorm(spatial_dim)

    def forward(self, u_t, h_t, adjacency):
        node_features = self.in_proj(torch.cat((u_t, h_t), dim=-1))
        mixed = self.graph_mix(node_features, adjacency)
        return self.norm(node_features + self.out_proj(mixed))
    
    
class DynamicGraphEvolution(nn.Module):
    def __init__(
        self,
        h_dim,
        u_dim,
        node_dim,
        graph_state_dim,
        attn_dim,
        dropout,
        graph_update_scale,
    ):
        super().__init__()
        self.graph_state_dim = graph_state_dim
        self.graph_update_scale = graph_update_scale
        self.node_encoder = nn.Sequential(
            nn.Linear(h_dim + u_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, graph_state_dim),
            nn.SiLU(),
        )
        self.graph_gru = nn.GRUCell(graph_state_dim, graph_state_dim)
        self.query_proj = nn.Linear(graph_state_dim, attn_dim)
        self.key_proj = nn.Linear(graph_state_dim, attn_dim)
        self.recv_gate = nn.Linear(graph_state_dim, 1)
        self.send_gate = nn.Linear(graph_state_dim, 1)
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, hidden, inputs, previous_adjacency, previous_state):
        batch_size, nodes, _ = hidden.shape
        encoded = self.node_encoder(torch.cat((hidden, inputs), dim=-1))
        state = self.graph_gru(
            encoded.reshape(batch_size * nodes, -1),
            previous_state.reshape(batch_size * nodes, -1),
        ).reshape(batch_size, nodes, self.graph_state_dim)
        logits = self.query_proj(state) @ self.key_proj(state).transpose(1, 2)
        candidate = torch.sigmoid(logits / math.sqrt(self.key_proj.out_features))
        gate = torch.sigmoid(
            self.recv_gate(state) + self.send_gate(state).transpose(1, 2) + self.gate_bias
        )
        adjacency = previous_adjacency + self.graph_update_scale * gate * (
            candidate - previous_adjacency
        )
        return adjacency, state


class GraphAwareDenoiseResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, dropout=0.1, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, condition, time_embedding, adjacency):
        block_input = hidden + self.cond_proj(condition) + self.time_proj(time_embedding).unsqueeze(1)
        adjacency = adjacency.clamp_min(0)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        messages = torch.bmm(adjacency, block_input)
        gate = self.self_proj(block_input) + self.msg_proj(messages)
        filter_part, gate_part = gate.chunk(2, dim=-1)
        activated = torch.tanh(filter_part) * torch.sigmoid(gate_part)
        residual, skip = self.out_proj(self.dropout(activated)).chunk(2, dim=-1)
        return (hidden + residual) / math.sqrt(2), skip


class GraphAwareDenoiseNet(nn.Module):
    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        hidden_dim: int,
        diff_step_emb_dim: int | None = None,
        dropout: float = 0.1,
        num_graph_blocks: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.diff_step_emb_dim = diff_step_emb_dim or hidden_dim
        self.x_proj = nn.Linear(x_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.diff_step_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_blocks = nn.ModuleList(
            GraphAwareDenoiseResidualBlock(hidden_dim, cond_dim, dropout, eps)
            for _ in range(num_graph_blocks)
        )
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x, diffusion_step, cond, adjacency):
        diffusion_step = torch.as_tensor(diffusion_step, device=x.device).reshape(-1)
        time_embedding = self.time_mlp(timestep_embedding(diffusion_step, self.diff_step_emb_dim))
        hidden = self.input_mlp(self.x_proj(x))
        skip_sum = 0
        for block in self.graph_blocks:
            hidden, skip = block(hidden, cond, time_embedding, adjacency)
            skip_sum += skip
        return self.out_mlp(F.silu(skip_sum / math.sqrt(len(self.graph_blocks))))
