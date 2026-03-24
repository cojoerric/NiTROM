import numpy as np
import pytest
import torch

from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.time_steppers.time_stepper import evolve, solve_ivp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 4  # state dimension
B = 3  # batch size
T0 = 0.0
TF = 0.5
DT_BASE = 1 / 500  # coarsest dt; each level halves it
N_LEVELS = 5  # number of refinement levels


def _make_model(rng):
    """Create a PolynomialModel with poly_comp=[1, 2] and random tensors."""
    A1 = torch.tensor(rng.standard_normal((N, N)), dtype=torch.float64)
    # Make A1 strongly stable (shift eigenvalues well to the left)
    A1 = A1 - 10.0 * torch.eye(N, dtype=torch.float64)
    A2 = torch.tensor(
        0.01 * rng.standard_normal((N, N, N)), dtype=torch.float64
    )
    model = PolynomialModel([1, 2], [A1, A2])
    model._generate_einsum_subscripts()
    return model


def _solve_at_tf(model, x0, dt, method):
    """
    Integrate the model from T0 to TF with step size dt using the
    evolve function directly and return the solution at TF.
    """
    nt = int(np.round((TF - T0) / dt))
    dt_actual = (TF - T0) / nt
    x = x0.clone()
    t = T0
    for _ in range(nt):
        x = evolve(model.evaluate_rhs, t, x, dt_actual, method)
        t += dt_actual
    return x


def _compute_successive_errors(model, x0, method):
    """
    Return arrays of dt values and successive-solution errors
    e_j = ||sol_j(tf) - sol_{j-1}(tf)||.
    """
    dts = [DT_BASE / (2 ** k) for k in range(N_LEVELS)]
    sols = []
    for dt in dts:
        sol = _solve_at_tf(model, x0, dt, method)
        sols.append(sol)

    errors = []
    for j in range(1, len(sols)):
        err = torch.linalg.norm(sols[j] - sols[j - 1]).item()
        errors.append(err)

    return dts, errors


def _check_order(errors, expected_order):
    """
    Given successive errors e_1, e_2, ..., check that
    e_j / e_{j+1} ≈ 2^p with 1% tolerance.
    """
    expected_ratio = 2 ** expected_order
    # Use last two ratios (most asymptotic) and allow 5% tolerance
    # because successive-difference convergence has higher-order bias
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        np.testing.assert_allclose(
            ratio, expected_ratio, rtol=1e-2,
            err_msg=f"Ratio = {ratio:.6f}, expected {expected_ratio:.1f}",
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    rng = np.random.default_rng(42)
    return _make_model(rng)


@pytest.fixture
def x0_unbatched():
    rng = np.random.default_rng(100)
    x0 = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
    return 0.01 * x0 / torch.linalg.vector_norm(x0)


@pytest.fixture
def x0_batched():
    rng = np.random.default_rng(200)
    x0 = torch.tensor(rng.standard_normal((B, N)), dtype=torch.float64)
    norms = torch.linalg.vector_norm(x0, dim=-1, keepdim=True)
    return 0.01 * x0 / norms


# ---------------------------------------------------------------------------
# RK4 convergence tests
# ---------------------------------------------------------------------------

class TestRK4Convergence:

    def test_unbatched(self, model, x0_unbatched):
        _, errors = _compute_successive_errors(
            model, x0_unbatched, "rk4"
        )
        _check_order(errors, expected_order=4)

    def test_batched(self, model, x0_batched):
        _, errors = _compute_successive_errors(
            model, x0_batched, "rk4"
        )
        _check_order(errors, expected_order=4)


# ---------------------------------------------------------------------------
# RK2 convergence tests
# ---------------------------------------------------------------------------

class TestRK2Convergence:

    def test_unbatched(self, model, x0_unbatched):
        _, errors = _compute_successive_errors(
            model, x0_unbatched, "rk2"
        )
        _check_order(errors, expected_order=2)

    def test_batched(self, model, x0_batched):
        _, errors = _compute_successive_errors(
            model, x0_batched, "rk2"
        )
        _check_order(errors, expected_order=2)


# ---------------------------------------------------------------------------
# Batched vs unbatched consistency
# ---------------------------------------------------------------------------

class TestBatchedConsistency:

    @pytest.mark.parametrize("method", ["rk4", "rk2"])
    def test_batched_matches_unbatched(self, model, method):
        """Each row of the batched solution should match the
        corresponding unbatched solve."""
        rng = np.random.default_rng(300)
        x0_batch = torch.tensor(
            rng.standard_normal((B, N)), dtype=torch.float64
        )
        norms = torch.linalg.vector_norm(x0_batch, dim=-1, keepdim=True)
        x0_batch = 0.01 * x0_batch / norms
        dt = 1e-3

        sol_batched = _solve_at_tf(model, x0_batch, dt, method)  # (B, n)

        for b in range(B):
            sol_single = _solve_at_tf(model, x0_batch[b], dt, method)  # (n,)
            np.testing.assert_allclose(
                sol_batched[b].numpy(), sol_single.numpy(), rtol=1e-12,
                err_msg=f"Batch index {b} mismatch for method={method}",
            )
