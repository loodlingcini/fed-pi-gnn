import logging
import math
import random
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import nn

from fed_pi_gnn.config import ExperimentConfig
from fed_pi_gnn.data import FormulationDataset, iter_client, loader
from fed_pi_gnn.federation import ClientUpdate, apply_trunk, clone_client, trunk_state, weighted_fedavg
from fed_pi_gnn.losses import JointObjective, LossWeights, fedprox_penalty
from fed_pi_gnn.model import FedPIGNN


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    model: FedPIGNN
    losses: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: ExperimentConfig) -> FedPIGNN:
    return FedPIGNN(config.atom_dim, config.formulation_dim, config.hidden_dim, config.layers, config.organs, config.time_points, config.clients, config.dropout)


def cosine_multiplier(round_index: int, rounds: int, warmup: int) -> float:
    if warmup > 0 and round_index < warmup:
        return (round_index + 1) / warmup
    progress = (round_index - warmup) / max(1, rounds - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_client(global_model: FedPIGNN, dataset: FormulationDataset, config: ExperimentConfig, client: int, round_index: int, device: torch.device) -> tuple[FedPIGNN, float]:
    local = clone_client(global_model).to(device)
    optimizer = torch.optim.AdamW(local.parameters(), lr=config.learning_rate * cosine_multiplier(round_index, config.rounds, config.warmup_rounds), weight_decay=config.weight_decay)
    objective = JointObjective(LossWeights(config.lambda_pbpk, config.lambda_stokes_einstein, config.lambda_gibbs, config.lambda_epr), config.temperature_kelvin, config.plasma_ph, config.endosomal_ph, config.half_life_hours).to(device)
    reference = {name: parameter.detach().clone() for name, parameter in global_model.named_parameters()}
    values: list[float] = []
    batches = loader(dataset, min(config.batch_size, len(dataset)), True)
    for _ in range(config.local_epochs):
        for step, batch in enumerate(batches):
            if step >= config.steps_per_round:
                break
            current = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = local(current)
            breakdown = objective(output, current, round_index >= config.rounds // 2)
            loss = breakdown.total + config.fedprox_mu * fedprox_penalty(local, reference)
            loss.backward()
            nn.utils.clip_grad_norm_(local.parameters(), config.gradient_clip)
            optimizer.step()
            values.append(float(loss.detach().item()))
    return cast(FedPIGNN, local), float(np.mean(values))


def train_federated(dataset: FormulationDataset, config: ExperimentConfig, device: torch.device | None = None) -> TrainingResult:
    set_seed(config.seed)
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = build_model(config).to(selected_device)
    losses: list[float] = []
    for round_index in range(config.rounds):
        updates: list[ClientUpdate] = []
        for client in range(config.clients):
            records = list(iter_client(dataset.records, client))
            if not records:
                continue
            local, loss = train_client(global_model, FormulationDataset(records), config, client, round_index, selected_device)
            updates.append(ClientUpdate(client, len(records), trunk_state(local), loss))
        apply_trunk(global_model, weighted_fedavg(updates))
        round_loss = float(np.average([update.loss for update in updates], weights=[update.examples for update in updates]))
        losses.append(round_loss)
        LOGGER.info("round=%d loss=%.6f", round_index + 1, round_loss)
    return TrainingResult(global_model, losses)
