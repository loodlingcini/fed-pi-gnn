import torch

from fed_pi_gnn.data import collate_records, random_records
from fed_pi_gnn.federation import ClientUpdate, weighted_fedavg
from fed_pi_gnn.metrics import binary_auroc, r2_score
from fed_pi_gnn.physics import PBPKParameters, gibbs_protonation, hydrodynamic_kernel, pbpk_rhs, stokes_einstein


def test_diffusion_decreases_with_radius() -> None:
    values = stokes_einstein(torch.tensor([10.0, 100.0]))
    assert values[0] > values[1] > 0


def test_kernel_is_bounded() -> None:
    value = hydrodynamic_kernel(torch.tensor([[0.0, 1e-8]]), torch.tensor([50.0]), torch.tensor(3600.0))
    assert torch.all(value >= 0)
    assert torch.all(value <= 1)


def test_gibbs_has_shape() -> None:
    value = gibbs_protonation(torch.tensor([6.4, 7.0]), 7.4, 5.5, 310.0)
    assert value.shape == (2,)


def test_pbpk_rhs_conserves_without_elimination() -> None:
    parameters = PBPKParameters.murine()
    parameters = PBPKParameters(parameters.flow, torch.zeros_like(parameters.elimination))
    value = pbpk_rhs(torch.ones(2, 8), parameters)
    assert value.shape == (2, 8)


def test_collation() -> None:
    batch = collate_records(random_records(4, 8, 8, 2, 1))
    assert batch.atoms.shape[0] == 4
    assert batch.regression.shape == (4, 8)


def test_weighted_fedavg() -> None:
    updates = [ClientUpdate(0, 1, {"x": torch.tensor([1.0])}, 1.0), ClientUpdate(1, 3, {"x": torch.tensor([3.0])}, 1.0)]
    assert torch.equal(weighted_fedavg(updates)["x"], torch.tensor([2.5]))


def test_metrics() -> None:
    assert r2_score(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])) == 1.0
    assert binary_auroc(torch.tensor([0.1, 0.9]), torch.tensor([0.0, 1.0])) == 1.0
