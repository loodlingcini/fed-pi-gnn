from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    seeds: int
    clients: int
    rounds: int
    local_epochs: int
    steps_per_round: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_rounds: int
    scheduler: str
    precision: str
    hidden_dim: int
    layers: int
    dropout: float
    atom_dim: int
    formulation_dim: int
    organs: int
    time_points: int
    fedprox_mu: float
    lambda_pbpk: float
    lambda_stokes_einstein: float
    lambda_gibbs: float
    lambda_epr: float
    temperature_kelvin: float
    viscosity_pascal_seconds: float
    plasma_ph: float
    endosomal_ph: float
    half_life_hours: float
    gradient_clip: float
    split: str
    split_ratios: tuple[float, float, float]
    bootstrap_samples: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ExperimentConfig":
        prepared = dict(value)
        prepared["split_ratios"] = tuple(float(x) for x in prepared["split_ratios"])
        return cls(**prepared)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return ExperimentConfig.from_mapping(value)

