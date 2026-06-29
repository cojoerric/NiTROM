import pytest
import torch
import torch.nn as nn

from nitrom.optimization import train
from nitrom.optimization.modules.base import InferenceModule


class DummyManifoldModule(InferenceModule):
    """A simple InferenceModule for testing manifold optimization."""

    def __init__(self, N=5, r=2):
        super().__init__()
        self.Phi = nn.Parameter(torch.randn(N, r, dtype=torch.float64))
        self.Psi = nn.Parameter(torch.randn(N, r, dtype=torch.float64))
        self.theta = nn.Parameter(torch.randn(r, r, dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        # Minimum at Phi=Psi=theta=0, but Phi, Psi are constrained to be orthonormal.
        # Under orthonormal constraints, the minimum is when parameters are orthonormal.
        return (self.Phi**2).sum() + (self.Psi**2).sum() + (self.theta**2).sum()

    def gradient(self) -> list[torch.Tensor]:
        # Analytical gradients of the forward cost
        return [2.0 * self.Phi, 2.0 * self.Psi, 2.0 * self.theta]


def test_manifold_types_validation():
    """Verify that train() raises ValueError for invalid parameter names or manifold types."""
    module = DummyManifoldModule()

    # Invalid parameter name
    with pytest.raises(ValueError, match="Parameter 'non_existent' specified in manifold_types does not exist"):
        train(module, n_epochs=1, manifold_types={"non_existent": "grassmann"})

    # Invalid manifold type
    with pytest.raises(ValueError, match="Manifold type for parameter 'Phi' must be one of"):
        train(module, n_epochs=1, manifold_types={"Phi": "invalid_manifold"})


@pytest.mark.parametrize("optimizer_type", ["adam", "sgd", "lbfgs"])
def test_manifold_optimization_properties(optimizer_type):
    """Verify that parameters stay orthonormal and gradients are projected correctly."""
    module = DummyManifoldModule(N=5, r=2)
    I_r = torch.eye(2, dtype=torch.float64)

    # Initial matrices are random and not orthonormal
    assert not torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-3)
    assert not torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-3)

    # Train for 5 epochs
    manifold_types = {"Phi": "grassmann", "Psi": "stiefel"}
    train(
        module,
        n_epochs=5,
        lr=0.01,
        optimizer_type=optimizer_type,
        print_every=1,
        manifold_types=manifold_types,
    )

    # 1. Check orthonormality constraints after training
    assert torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-12)
    assert torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-12)

    # 2. Check that the assigned gradients are in the respective tangent spaces
    assert module.Phi.grad is not None
    assert module.Psi.grad is not None

    # Grassmann constraint: Phi.T @ grad_Phi = 0
    assert torch.allclose(module.Phi.T @ module.Phi.grad, torch.zeros(2, 2, dtype=torch.float64), atol=1e-7)

    # Stiefel constraint: Psi.T @ grad_Psi + grad_Psi.T @ Psi = 0 (skew-symmetric)
    skew = module.Psi.T @ module.Psi.grad
    assert torch.allclose(skew + skew.T, torch.zeros(2, 2, dtype=torch.float64), atol=1e-7)


def test_euclidean_behaves_normally():
    """Verify that specifying 'euclidean' explicitly behaves like the default."""
    module = DummyManifoldModule(N=5, r=2)
    I_r = torch.eye(2, dtype=torch.float64)

    # Mark all as Euclidean
    train(
        module,
        n_epochs=2,
        lr=0.01,
        optimizer_type="sgd",
        manifold_types={"Phi": "euclidean", "Psi": "euclidean"},
    )

    # Since they are Euclidean, they should not be orthonormalized
    assert not torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-3)
    assert not torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-3)


class NitromModule(InferenceModule):
    """A mock NitromModule to test automatic parameter detection."""

    def __init__(self, N=5, r=2):
        super().__init__()
        self.Phi = nn.Parameter(torch.randn(N, r, dtype=torch.float64))
        self.Psi = nn.Parameter(torch.randn(N, r, dtype=torch.float64))
        self.theta = nn.Parameter(torch.randn(r, r, dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        return (self.Phi**2).sum() + (self.Psi**2).sum() + (self.theta**2).sum()

    def gradient(self) -> list[torch.Tensor]:
        return [2.0 * self.Phi, 2.0 * self.Psi, 2.0 * self.theta]


def test_nitrom_automatic_manifolds():
    """Verify that if type(model).__name__ == 'NitromModule', it automatically sets default manifolds."""
    module = NitromModule(N=5, r=2)
    I_r = torch.eye(2, dtype=torch.float64)

    # Initially not orthonormal
    assert not torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-3)
    assert not torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-3)

    # Train without specifying manifold_types (should default to Phi: grassmann, Psi: stiefel)
    train(
        module,
        n_epochs=3,
        lr=0.01,
        optimizer_type="sgd",
    )

    # Verify they were retracted to be orthonormal
    assert torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-12)
    assert torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-12)

    # Verify that gradients are in their respective tangent spaces
    assert module.Phi.grad is not None
    assert module.Psi.grad is not None
    assert torch.allclose(module.Phi.T @ module.Phi.grad, torch.zeros(2, 2, dtype=torch.float64), atol=1e-7)

    skew = module.Psi.T @ module.Psi.grad
    assert torch.allclose(skew + skew.T, torch.zeros(2, 2, dtype=torch.float64), atol=1e-7)


def test_nitrom_user_priority():
    """Verify that user-provided manifold_types override the NitromModule defaults."""
    module = NitromModule(N=5, r=2)
    I_r = torch.eye(2, dtype=torch.float64)

    # Override Phi to be Euclidean, while leaving Psi as default (Stiefel)
    train(
        module,
        n_epochs=3,
        lr=0.01,
        optimizer_type="sgd",
        manifold_types={"Phi": "euclidean"},
    )

    # Phi should remain Euclidean (not orthonormal)
    assert not torch.allclose(module.Phi.T @ module.Phi, I_r, atol=1e-3)
    # Psi should have been retracted to Stiefel (orthonormal)
    assert torch.allclose(module.Psi.T @ module.Psi, I_r, atol=1e-12)

    # Psi's gradient should be in Stiefel tangent space
    assert module.Psi.grad is not None
    skew = module.Psi.T @ module.Psi.grad
    assert torch.allclose(skew + skew.T, torch.zeros(2, 2, dtype=torch.float64), atol=1e-7)

