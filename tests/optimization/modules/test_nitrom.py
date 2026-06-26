"""Finite-difference gradient checks for NitromModule (polynomial and GAS ROMs).

Ambient dimension N = 5, latent dimension r = 2.  The module's analytic gradient
is a continuous adjoint, so the comparison is against a finite difference of the
forward solve at a fairly fine sub-step resolution (the two agree to
discretization accuracy).
"""

import torch

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections.linear_projection import LinearProjection
from nitrom.roms.param_registry import ParamRegistry

from ._gradcheck import (
    DTYPE,
    Data,
    LinearOutputFOM,
    assert_grad_close,
    finite_diff_grad,
    random_basis,
)

N, R, NTRAJ, NT, M, NO = 5, 2, 2, 4, 1, 3


def _data(seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(NTRAJ, N, NT, generator=g, dtype=DTYPE)
    weights = 1.0 + 0.5 * torch.rand(NTRAJ, generator=g, dtype=DTYPE)
    forcing_fns = [
        (lambda t, c=0.5 * (i + 1): torch.tensor([c], dtype=DTYPE))
        for i in range(NTRAJ)
    ]
    return Data(X, weights=weights, forcing_fns=forcing_fns)


def _projection() -> LinearProjection:
    # Oblique projection (Psi != Phi) so the encoder/decoder gradients for both
    # bases are exercised; both are trainable in NiTROM.
    Phi = random_basis(N, R, seed=1)
    Psi = random_basis(N, R, seed=2)
    return LinearProjection([Phi, Psi])


def _fom(seed: int = 4) -> LinearOutputFOM:
    g = torch.Generator().manual_seed(seed)
    return LinearOutputFOM(torch.randn(NO, N, generator=g, dtype=DTYPE))


def _module(model) -> NitromModule:
    registry = ParamRegistry(model, _projection())
    return NitromModule(
        _data(), registry, fom=_fom(), reg=0.01,
        n_substeps=100, time_stepper="rk4", n_leggauss=5,
    )


def test_gradient_polynomial_rom():
    g = torch.Generator().manual_seed(2)
    tensors = [
        0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),
        0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),
        0.3 * torch.randn(R, M, generator=g, dtype=DTYPE),
    ]
    model = PolynomialModel(
        R, [1, 2], dtype=DTYPE, tensors=tensors,
        forcing_config={"forcing_exists": True, "m": M},
    )
    module = _module(model)
    assert_grad_close(
        module.gradient(), finite_diff_grad(module), rtol=1e-4, atol=1e-6
    )


def test_gradient_gas_rom():
    g = torch.Generator().manual_seed(3)
    eye = torch.eye(R, dtype=DTYPE)
    gas_params = [
        0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),         # K
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),    # R
        eye + 0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),    # Q
        0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),       # S
        0.3 * torch.randn(R, M, generator=g, dtype=DTYPE),         # B
    ]
    model = GasPolynomialModel(
        R, [1, 2], dtype=DTYPE, gas_params=gas_params,
        forcing_config={"forcing_exists": True, "m": M},
    )
    module = _module(model)
    assert_grad_close(
        module.gradient(), finite_diff_grad(module), rtol=1e-4, atol=1e-6
    )
