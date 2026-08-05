from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fed_pi_gnn.data import GraphBatch
from fed_pi_gnn.physics import hydrodynamic_kernel


@dataclass(frozen=True)
class ModelOutput:
    regression: Tensor
    logits: Tensor
    concentrations: Tensor
    learned_edges: Tensor
    physical_edges: Tensor
    predicted_pka: Tensor
    times: Tensor


class AtomEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, atoms: Tensor) -> Tensor:
        return cast(Tensor, self.normalization(F.gelu(self.projection(atoms))))


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim**-0.5

    def forward(self, states: Tensor, adjacency: Tensor) -> Tensor:
        scores = torch.matmul(self.query(states), self.key(states).transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(adjacency <= 0, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        messages = torch.matmul(weights, self.value(states))
        return cast(Tensor, self.normalization(states + self.dropout(self.output(messages))))


class TimeEncoding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        frequencies = torch.exp(torch.linspace(0.0, 8.0, hidden_dim // 2) * -1.0)
        self.register_buffer("frequencies", frequencies)

    def forward(self, times: Tensor) -> Tensor:
        phases = times.unsqueeze(-1) * cast(Tensor, self.frequencies)
        encoded = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
        return F.pad(encoded, (0, self.hidden_dim - encoded.shape[-1]))


class FedPIGNN(nn.Module):
    def __init__(self, atom_dim: int, formulation_dim: int, hidden_dim: int, layers: int, organs: int, time_points: int, clients: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.atom_encoder = AtomEncoder(atom_dim, hidden_dim)
        self.layers = nn.ModuleList(GraphAttentionLayer(hidden_dim, dropout) for _ in range(layers))
        self.formulation_encoder = nn.Sequential(nn.Linear(formulation_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.organ_embeddings = nn.Parameter(torch.randn(organs, hidden_dim) * 0.02)
        self.organ_positions = nn.Parameter(torch.randn(organs, 3) * 0.02)
        self.edge_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.regression_head = nn.Linear(hidden_dim, 1)
        self.classification_heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(clients))
        self.pka_head = nn.Linear(hidden_dim, 1)
        self.time_encoding = TimeEncoding(hidden_dim)
        self.concentration_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Softplus())
        self.time_points = time_points
        self.tau = nn.Parameter(torch.tensor(3600.0))

    def encode(self, batch: GraphBatch) -> Tensor:
        states = self.atom_encoder(batch.atoms)
        for layer in self.layers:
            states = layer(states, batch.adjacency)
        atom_mask = batch.adjacency.sum(dim=-1, keepdim=True).gt(0)
        molecule = (states * atom_mask).sum(dim=1) / atom_mask.sum(dim=1).clamp_min(1)
        return cast(Tensor, molecule + self.formulation_encoder(batch.formulation))

    def forward(self, batch: GraphBatch) -> ModelOutput:
        molecule = self.encode(batch)
        batch_size = molecule.shape[0]
        organs = self.organ_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        molecule_nodes = molecule.unsqueeze(1).expand_as(organs)
        joint = torch.cat((molecule_nodes, organs), dim=-1)
        learned_edges = self.edge_gate(joint).squeeze(-1)
        distance = torch.cdist(torch.zeros(batch_size, 1, 3, device=molecule.device), self.organ_positions.unsqueeze(0).expand(batch_size, -1, -1)).squeeze(1)
        physical_edges = hydrodynamic_kernel(distance * 1e-6, batch.radius_nm, self.tau)
        organ_states = organs + molecule_nodes * physical_edges.unsqueeze(-1)
        regression = self.regression_head(organ_states).squeeze(-1)
        logits = torch.empty_like(regression)
        for client, head in enumerate(self.classification_heads):
            mask = batch.clients == client
            if bool(mask.any()):
                logits[mask] = head(organ_states[mask]).squeeze(-1)
        predicted_pka = self.pka_head(molecule).squeeze(-1)
        times = torch.linspace(0.0, 24.0, self.time_points, device=molecule.device, requires_grad=True)
        temporal = self.time_encoding(times)
        field = organ_states.unsqueeze(2) + temporal.view(1, 1, self.time_points, -1)
        concentrations = self.concentration_head(field).squeeze(-1).transpose(1, 2)
        return ModelOutput(regression, logits, concentrations, learned_edges, physical_edges, predicted_pka, times)
