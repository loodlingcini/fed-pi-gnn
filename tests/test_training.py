from pathlib import Path

import torch

from fed_pi_gnn.config import load_config
from fed_pi_gnn.data import FormulationDataset, random_records
from fed_pi_gnn.evaluation import evaluate
from fed_pi_gnn.training import train_federated


ROOT = Path(__file__).parents[1]


def test_training_pipeline() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    dataset = FormulationDataset(random_records(16, config.atom_dim, config.organs, config.clients, config.seed))
    result = train_federated(dataset, config, torch.device("cpu"))
    assert len(result.losses) == 1
    metrics = evaluate(result.model, dataset, 4, torch.device("cpu"))
    assert metrics.mse >= 0.0

