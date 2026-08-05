from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ClientUpdate:
    client: int
    examples: int
    state: dict[str, Tensor]
    loss: float


def trunk_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items() if "classification_heads" not in name}


def weighted_fedavg(updates: Iterable[ClientUpdate]) -> dict[str, Tensor]:
    materialized = list(updates)
    if not materialized:
        raise ValueError("at least one client update is required")
    total = sum(update.examples for update in materialized)
    if total <= 0:
        raise ValueError("client example count must be positive")
    names = materialized[0].state.keys()
    aggregated: dict[str, Tensor] = {}
    for name in names:
        value = materialized[0].state[name].new_zeros(materialized[0].state[name].shape)
        for update in materialized:
            value = value + update.state[name] * (update.examples / total)
        aggregated[name] = value
    return aggregated


def apply_trunk(model: nn.Module, state: dict[str, Tensor]) -> None:
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current)


def clone_client(global_model: nn.Module) -> nn.Module:
    return deepcopy(global_model)


def parameter_drift(local: nn.Module, global_model: nn.Module) -> float:
    global_values = dict(global_model.named_parameters())
    squared = 0.0
    for name, parameter in local.named_parameters():
        if "classification_heads" not in name:
            squared += float((parameter.detach() - global_values[name].detach()).square().sum().item())
    return float(squared**0.5)


def atomic_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, round_index: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "round": round_index, "seed": seed, "rng": torch.get_rng_state()}, temporary)
    temporary.replace(path)


def restore_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(value["model"])
    optimizer.load_state_dict(value["optimizer"])
    torch.set_rng_state(value["rng"])
    return int(value["round"]), int(value["seed"])
