from dataclasses import dataclass

import torch
from torch import Tensor, nn
from typing import cast


BOLTZMANN = 1.380649e-23
GAS_CONSTANT = 8.31446261815324


@dataclass(frozen=True)
class PBPKParameters:
    flow: Tensor
    elimination: Tensor

    @classmethod
    def murine(cls, device: torch.device | None = None) -> "PBPKParameters":
        flow = torch.tensor(
            [
                [0.0, 0.08, 0.03, 0.02, 0.04, 0.03, 0.19, 0.01],
                [0.08, 0.0, 0.02, 0.01, 0.01, 0.01, 0.08, 0.01],
                [0.03, 0.02, 0.0, 0.02, 0.02, 0.03, 0.18, 0.01],
                [0.02, 0.01, 0.02, 0.0, 0.01, 0.01, 0.05, 0.01],
                [0.04, 0.01, 0.02, 0.01, 0.0, 0.02, 0.12, 0.01],
                [0.03, 0.01, 0.03, 0.01, 0.02, 0.0, 0.15, 0.01],
                [0.19, 0.08, 0.18, 0.05, 0.12, 0.15, 0.0, 0.08],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.08, 0.0],
            ],
            device=device,
        )
        elimination = torch.tensor([0.035, 0.025, 0.018, 0.012, 0.05, 0.02, 0.08, 0.01], device=device)
        return cls(flow, elimination)


def stokes_einstein(radius_nm: Tensor, temperature: float = 310.0, viscosity: float = 0.0012) -> Tensor:
    radius_m = radius_nm.clamp_min(1e-6) * 1e-9
    return BOLTZMANN * temperature / (6.0 * torch.pi * viscosity * radius_m)


def hydrodynamic_kernel(distance: Tensor, radius_nm: Tensor, tau: Tensor, temperature: float = 310.0, viscosity: float = 0.0012) -> Tensor:
    diffusion = stokes_einstein(radius_nm, temperature, viscosity)
    while diffusion.ndim < distance.ndim:
        diffusion = diffusion.unsqueeze(-1)
    scale = (2.0 * diffusion * tau.abs().clamp_min(1e-9)).clamp_min(1e-30)
    return torch.exp(-(distance.square() / scale).clamp_max(80.0))


def gibbs_protonation(pka: Tensor, plasma_ph: float, endosomal_ph: float, temperature: float) -> Tensor:
    protonation = torch.sigmoid(torch.log(torch.tensor(10.0, device=pka.device)) * (pka - endosomal_ph))
    reference = torch.sigmoid(torch.log(torch.tensor(10.0, device=pka.device)) * (pka - plasma_ph))
    return GAS_CONSTANT * temperature * (protonation - reference)


def pbpk_rhs(concentrations: Tensor, parameters: PBPKParameters) -> Tensor:
    incoming = torch.einsum("ij,...j->...i", parameters.flow, concentrations)
    outgoing = concentrations * parameters.flow.sum(dim=0)
    return incoming - outgoing - parameters.elimination * concentrations


def temporal_derivative(concentrations: Tensor, times: Tensor) -> Tensor:
    if times.numel() < 2:
        return torch.zeros_like(concentrations)
    return torch.gradient(concentrations, spacing=(times,), dim=(1,), edge_order=1)[0]


class PBPKResidual(nn.Module):
    def __init__(self, parameters: PBPKParameters) -> None:
        super().__init__()
        self.register_buffer("flow", parameters.flow)
        self.register_buffer("elimination", parameters.elimination)

    def forward(self, concentrations: Tensor, times: Tensor) -> Tensor:
        derivative = temporal_derivative(concentrations, times)
        parameters = PBPKParameters(cast(Tensor, self.flow), cast(Tensor, self.elimination))
        residual = derivative - pbpk_rhs(concentrations, parameters)
        return residual.square().mean()


def epr_clearance(concentrations: Tensor, times: Tensor, half_life: float, plasma_index: int = 6) -> Tensor:
    plasma = concentrations[..., plasma_index]
    initial = plasma[..., :1] if plasma.ndim > 1 else plasma[:1]
    envelope = initial * torch.pow(2.0, -times / half_life)
    return torch.relu(plasma - envelope).square().mean()
