from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fed_pi_gnn.data import GraphBatch
from fed_pi_gnn.model import ModelOutput
from fed_pi_gnn.physics import PBPKParameters, PBPKResidual, epr_clearance, gibbs_protonation


@dataclass(frozen=True)
class LossWeights:
    pbpk: float
    stokes_einstein: float
    gibbs: float
    epr: float


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    supervised: Tensor
    pbpk: Tensor
    stokes_einstein: Tensor
    gibbs: Tensor
    epr: Tensor


class JointObjective(nn.Module):
    def __init__(self, weights: LossWeights, temperature: float = 310.0, plasma_ph: float = 7.4, endosomal_ph: float = 5.5, half_life: float = 2.0) -> None:
        super().__init__()
        self.weights = weights
        self.temperature = temperature
        self.plasma_ph = plasma_ph
        self.endosomal_ph = endosomal_ph
        self.half_life = half_life
        self.pbpk = PBPKResidual(PBPKParameters.murine())

    def forward(self, output: ModelOutput, batch: GraphBatch, clearance_active: bool = True) -> LossBreakdown:
        regression = F.mse_loss(output.regression, batch.regression)
        classification = F.binary_cross_entropy_with_logits(output.logits, batch.classification)
        supervised = regression + classification
        pbpk = self.pbpk(output.concentrations, output.times)
        stokes = F.mse_loss(output.learned_edges, output.physical_edges)
        predicted = gibbs_protonation(output.predicted_pka, self.plasma_ph, self.endosomal_ph, self.temperature)
        reference = gibbs_protonation(batch.pka, self.plasma_ph, self.endosomal_ph, self.temperature)
        gibbs = F.mse_loss(predicted, reference)
        epr = epr_clearance(output.concentrations, output.times, self.half_life) if clearance_active else output.regression.new_zeros(())
        total = supervised + self.weights.pbpk * pbpk + self.weights.stokes_einstein * stokes + self.weights.gibbs * gibbs + self.weights.epr * epr
        return LossBreakdown(total, supervised, pbpk, stokes, gibbs, epr)


def fedprox_penalty(local: nn.Module, global_parameters: dict[str, Tensor]) -> Tensor:
    parameters = list(local.parameters())
    if not parameters:
        return torch.tensor(0.0)
    penalty = parameters[0].new_zeros(())
    for name, parameter in local.named_parameters():
        if "classification_heads" not in name:
            penalty = penalty + (parameter - global_parameters[name]).square().sum()
    return 0.5 * penalty

