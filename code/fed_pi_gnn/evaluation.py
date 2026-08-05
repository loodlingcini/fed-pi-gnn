from dataclasses import dataclass

import torch

from fed_pi_gnn.data import FormulationDataset, loader
from fed_pi_gnn.metrics import binary_auroc, brier_score, expected_calibration_error, mean_squared_error, r2_score
from fed_pi_gnn.model import FedPIGNN


@dataclass(frozen=True)
class EvaluationResult:
    r2: float
    mse: float
    auroc: tuple[float, ...]
    brier: float
    ece: float


@torch.no_grad()
def evaluate(model: FedPIGNN, dataset: FormulationDataset, batch_size: int, device: torch.device) -> EvaluationResult:
    model.eval()
    predictions = []
    targets = []
    probabilities = []
    labels = []
    for batch in loader(dataset, batch_size, False):
        current = batch.to(device)
        output = model(current)
        predictions.append(output.regression.cpu())
        targets.append(current.regression.cpu())
        probabilities.append(output.logits.sigmoid().cpu())
        labels.append(current.classification.cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    probability = torch.cat(probabilities)
    label = torch.cat(labels)
    auroc = tuple(binary_auroc(probability[:, index], label[:, index]) for index in range(label.shape[1]))
    return EvaluationResult(r2_score(prediction, target), mean_squared_error(prediction, target), auroc, brier_score(probability, label), expected_calibration_error(probability, label))

