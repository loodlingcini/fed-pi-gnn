from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def mean_squared_error(prediction: Tensor, target: Tensor) -> float:
    return float((prediction - target).square().mean().item())


def r2_score(prediction: Tensor, target: Tensor) -> float:
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum()
    return float((1.0 - residual / total.clamp_min(torch.finfo(target.dtype).eps)).item())


def brier_score(probabilities: Tensor, target: Tensor) -> float:
    return float((probabilities - target).square().mean().item())


def binary_auroc(scores: Tensor, target: Tensor) -> float:
    positives = scores[target == 1]
    negatives = scores[target == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).float()
    ties = (positives[:, None] == negatives[None, :]).float() * 0.5
    return float((comparisons + ties).mean().item())


def expected_calibration_error(probabilities: Tensor, target: Tensor, bins: int = 15) -> float:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    result = probabilities.new_zeros(())
    for index in range(bins):
        mask = (probabilities > edges[index]) & (probabilities <= edges[index + 1])
        if bool(mask.any()):
            result = result + mask.float().mean() * (probabilities[mask].mean() - target[mask].mean()).abs()
    return float(result.item())


def retention_ratio(federated: float, centralized: float) -> float:
    if centralized == 0.0:
        raise ValueError("centralized score must be nonzero")
    return federated / centralized


def worst_subgroup(scores: dict[str, float]) -> tuple[float, float]:
    if not scores:
        raise ValueError("subgroup scores cannot be empty")
    values = list(scores.values())
    return min(values), max(values) - min(values)


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float


def paired_bootstrap(first: np.ndarray, second: np.ndarray, samples: int = 10000, seed: int = 1) -> BootstrapInterval:
    if first.shape != second.shape:
        raise ValueError("paired arrays must have identical shapes")
    generator = np.random.default_rng(seed)
    differences = first - second
    estimates = np.empty(samples)
    for index in range(samples):
        selection = generator.integers(0, differences.size, differences.size)
        estimates[index] = differences[selection].mean()
    return BootstrapInterval(float(differences.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)))


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(p_values)
    decisions = [False] * len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (len(p_values) - rank):
            decisions[index] = True
        else:
            break
    return decisions

