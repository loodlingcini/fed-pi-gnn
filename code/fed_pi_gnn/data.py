from dataclasses import dataclass
from hashlib import sha256
from typing import Iterator, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class FormulationRecord:
    identifier: str
    smiles: str
    atom_features: Tensor
    adjacency: Tensor
    formulation: Tensor
    radius_nm: float
    pka: float
    regression: Tensor
    classification: Tensor
    client: int


@dataclass(frozen=True)
class GraphBatch:
    atoms: Tensor
    adjacency: Tensor
    formulation: Tensor
    radius_nm: Tensor
    pka: Tensor
    regression: Tensor
    classification: Tensor
    clients: Tensor

    def to(self, device: torch.device) -> "GraphBatch":
        return GraphBatch(
            self.atoms.to(device),
            self.adjacency.to(device),
            self.formulation.to(device),
            self.radius_nm.to(device),
            self.pka.to(device),
            self.regression.to(device),
            self.classification.to(device),
            self.clients.to(device),
        )


class FormulationDataset(Dataset[FormulationRecord]):
    def __init__(self, records: list[FormulationRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FormulationRecord:
        return self.records[index]


def canonical_smiles(value: str) -> str:
    return "".join(value.split())


def chemistry_hash(value: str) -> str:
    return sha256(canonical_smiles(value).encode("utf-8")).hexdigest()


def collate_records(records: list[FormulationRecord]) -> GraphBatch:
    maximum = max(record.atom_features.shape[0] for record in records)
    feature_dim = records[0].atom_features.shape[1]
    atoms = torch.zeros(len(records), maximum, feature_dim)
    adjacency = torch.zeros(len(records), maximum, maximum)
    for index, record in enumerate(records):
        size = record.atom_features.shape[0]
        atoms[index, :size] = record.atom_features
        adjacency[index, :size, :size] = record.adjacency
    return GraphBatch(
        atoms,
        adjacency,
        torch.stack([record.formulation for record in records]),
        torch.tensor([record.radius_nm for record in records]),
        torch.tensor([record.pka for record in records]),
        torch.stack([record.regression for record in records]),
        torch.stack([record.classification for record in records]),
        torch.tensor([record.client for record in records], dtype=torch.long),
    )


def loader(dataset: FormulationDataset, batch_size: int, shuffle: bool) -> DataLoader[GraphBatch]:
    return cast(DataLoader[GraphBatch], DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_records))


def random_records(count: int, atom_dim: int, organs: int, clients: int, seed: int) -> list[FormulationRecord]:
    generator = torch.Generator().manual_seed(seed)
    records: list[FormulationRecord] = []
    for index in range(count):
        atom_count = int(torch.randint(4, 12, (1,), generator=generator).item())
        atoms = torch.randn(atom_count, atom_dim, generator=generator)
        adjacency = torch.eye(atom_count)
        for position in range(atom_count - 1):
            adjacency[position, position + 1] = 1.0
            adjacency[position + 1, position] = 1.0
        formulation = torch.rand(8, generator=generator)
        regression = torch.sigmoid(formulation.mean() + torch.randn(organs, generator=generator) * 0.05)
        classification = (regression > regression.quantile(0.75)).float()
        records.append(
            FormulationRecord(
                str(index),
                "C" * atom_count,
                atoms,
                adjacency,
                formulation,
                40.0 + float(torch.rand(1, generator=generator).item()) * 80.0,
                5.5 + float(torch.rand(1, generator=generator).item()) * 2.0,
                regression,
                classification,
                index % clients,
            )
        )
    return records


def split_indices(hashes: list[str], ratios: tuple[float, float, float], seed: int) -> tuple[list[int], list[int], list[int]]:
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to one")
    order = np.random.default_rng(seed).permutation(len(hashes)).tolist()
    train_end = int(len(order) * ratios[0])
    validation_end = train_end + int(len(order) * ratios[1])
    return order[:train_end], order[train_end:validation_end], order[validation_end:]


def iter_client(records: list[FormulationRecord], client: int) -> Iterator[FormulationRecord]:
    return (record for record in records if record.client == client)
