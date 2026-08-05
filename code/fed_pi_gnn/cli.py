import argparse
import json
import logging
from pathlib import Path

import torch

from fed_pi_gnn.config import load_config
from fed_pi_gnn.data import FormulationDataset, random_records
from fed_pi_gnn.evaluation import evaluate
from fed_pi_gnn.training import build_model, train_federated


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="fed-pi-gnn")
    value.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    value.add_argument("--output", type=Path, default=Path("runs/model.pt"))
    value.add_argument("--records", type=int, default=256)
    return value


def train_main() -> None:
    logging.basicConfig(level=logging.INFO)
    arguments = parser().parse_args()
    config = load_config(arguments.config)
    dataset = FormulationDataset(random_records(arguments.records, config.atom_dim, config.organs, config.clients, config.seed))
    result = train_federated(dataset, config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": result.model.state_dict(), "config": config.__dict__, "losses": result.losses}, arguments.output)


def evaluate_main() -> None:
    arguments = parser().parse_args()
    config = load_config(arguments.config)
    dataset = FormulationDataset(random_records(arguments.records, config.atom_dim, config.organs, config.clients, config.seed))
    model = build_model(config)
    value = torch.load(arguments.output, map_location="cpu", weights_only=True)
    model.load_state_dict(value["model"])
    result = evaluate(model, dataset, config.batch_size, torch.device("cpu"))
    print(json.dumps(result.__dict__))


def infer_main() -> None:
    evaluate_main()

