"""Tests for the LinearProjection class."""

import pytest
import numpy as np
import torch

from nitrom.projections.linear_projection import LinearProjection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 10  # full-space dimension
R = 3   # reduced-space dimension
M = 5   # batch size


def _random_bases(n, r, seed=42):
    """Generate random full-rank trial (Phi) and test (Psi) bases."""
    rng = np.random.default_rng(seed)
    Phi = torch.tensor(rng.standard_normal((n, r)), dtype=torch.float64)
    Psi = torch.tensor(rng.standard_normal((n, r)), dtype=torch.float64)
    return Phi, Psi


@pytest.fixture
def proj():
    Phi, Psi = _random_bases(N, R)
    return LinearProjection(Phi, Psi)


# ---------------------------------------------------------------------------
# encode tests
# ---------------------------------------------------------------------------

class TestEncode:

    def test_unbatched_shape(self, proj):
        q = torch.randn(N, dtype=torch.float64)
        z = proj.encode(q)
        assert z.shape == (R,)

    def test_batched_shape(self, proj):
        q = torch.randn(M, N, dtype=torch.float64)
        z = proj.encode(q)
        assert z.shape == (M, R)

    def test_unbatched_value(self, proj):
        """encode(q) = Psi^T q"""
        q = torch.randn(N, dtype=torch.float64)
        z = proj.encode(q)
        expected = proj.Psi.T @ q
        np.testing.assert_allclose(z.numpy(), expected.numpy(), rtol=1e-12)

    def test_batched_value(self, proj):
        """Each row of the batch should match the unbatched result."""
        q = torch.randn(M, N, dtype=torch.float64)
        z = proj.encode(q)
        for j in range(M):
            expected_j = proj.Psi.T @ q[j]
            np.testing.assert_allclose(z[j].numpy(), expected_j.numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# decode tests
# ---------------------------------------------------------------------------

class TestDecode:

    def test_unbatched_shape(self, proj):
        z = torch.randn(R, dtype=torch.float64)
        q = proj.decode(z)
        assert q.shape == (N,)

    def test_batched_shape(self, proj):
        z = torch.randn(M, R, dtype=torch.float64)
        q = proj.decode(z)
        assert q.shape == (M, N)

    def test_unbatched_value(self, proj):
        """decode(z) = Phi S z where S = (Psi^T Phi)^{-1}"""
        z = torch.randn(R, dtype=torch.float64)
        q = proj.decode(z)
        S = torch.linalg.inv(proj.Psi.T @ proj.Phi)
        expected = proj.Phi @ (S @ z)
        np.testing.assert_allclose(q.numpy(), expected.numpy(), rtol=1e-12)

    def test_batched_value(self, proj):
        z = torch.randn(M, R, dtype=torch.float64)
        q = proj.decode(z)
        S = torch.linalg.inv(proj.Psi.T @ proj.Phi)
        for j in range(M):
            expected_j = proj.Phi @ (S @ z[j])
            np.testing.assert_allclose(q[j].numpy(), expected_j.numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# encode-decode consistency tests
# ---------------------------------------------------------------------------

class TestEncodeDecode:

    def test_decode_encode_is_projection(self, proj):
        """
        For q in the column space of Phi, decode(encode(q)) should recover q
        (up to the oblique projection).
        Specifically: Phi S Psi^T (Phi c) = Phi S (Psi^T Phi) c = Phi c = q.
        """
        c = torch.randn(R, dtype=torch.float64)
        q = proj.Phi @ c  # q lives in col(Phi)
        z = proj.encode(q)
        q_reconstructed = proj.decode(z)
        np.testing.assert_allclose(
            q_reconstructed.numpy(), q.numpy(), rtol=1e-10
        )

    def test_decode_encode_is_projection_batched(self, proj):
        C = torch.randn(M, R, dtype=torch.float64)
        Q = (proj.Phi @ C.T).T  # (M, N), each row in col(Phi)
        Z = proj.encode(Q)
        Q_reconstructed = proj.decode(Z)
        np.testing.assert_allclose(
            Q_reconstructed.numpy(), Q.numpy(), rtol=1e-10
        )

    def test_encode_decode_is_identity_in_reduced_space(self, proj):
        """encode(decode(z)) = Psi^T Phi S z = (Psi^T Phi)(Psi^T Phi)^{-1} z = z"""
        z = torch.randn(R, dtype=torch.float64)
        z_roundtrip = proj.encode(proj.decode(z))
        np.testing.assert_allclose(
            z_roundtrip.numpy(), z.numpy(), rtol=1e-10
        )

    def test_encode_decode_is_identity_in_reduced_space_batched(self, proj):
        Z = torch.randn(M, R, dtype=torch.float64)
        Z_roundtrip = proj.encode(proj.decode(Z))
        np.testing.assert_allclose(
            Z_roundtrip.numpy(), Z.numpy(), rtol=1e-10
        )

    def test_projection_is_idempotent(self, proj):
        """Applying decode(encode(.)) twice should give the same result."""
        q = torch.randn(N, dtype=torch.float64)
        q1 = proj.decode(proj.encode(q))
        q2 = proj.decode(proj.encode(q1))
        np.testing.assert_allclose(q2.numpy(), q1.numpy(), rtol=1e-10)


# ---------------------------------------------------------------------------
# Orthogonal special case: Phi = Psi
# ---------------------------------------------------------------------------

class TestOrthogonalCase:

    def test_orthogonal_bases(self):
        """When Phi = Psi and columns are orthonormal, S = I."""
        rng = np.random.default_rng(99)
        A = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Phi, _ = torch.linalg.qr(A)
        proj = LinearProjection(Phi, Phi)

        np.testing.assert_allclose(
            proj.S.numpy(), np.eye(R), atol=1e-12
        )

        # decode(encode(q)) = Phi Phi^T q (orthogonal projection)
        q = torch.randn(N, dtype=torch.float64)
        q_proj = proj.decode(proj.encode(q))
        expected = Phi @ (Phi.T @ q)
        np.testing.assert_allclose(q_proj.numpy(), expected.numpy(), rtol=1e-10)


# ---------------------------------------------------------------------------
# update tests
# ---------------------------------------------------------------------------

class TestUpdate:

    def test_update_changes_bases(self, proj):
        Phi_old = proj.Phi.clone()
        Psi_old = proj.Psi.clone()

        Phi_new, Psi_new = _random_bases(N, R, seed=999)
        proj.update([Phi_new, Psi_new])

        assert torch.equal(proj.Phi, Phi_new)
        assert torch.equal(proj.Psi, Psi_new)
        assert not torch.equal(proj.Phi, Phi_old)
        assert not torch.equal(proj.Psi, Psi_old)

    def test_update_recomputes_S(self, proj):
        Phi_new, Psi_new = _random_bases(N, R, seed=888)
        proj.update([Phi_new, Psi_new])

        expected_S = torch.linalg.inv(Psi_new.T @ Phi_new)
        np.testing.assert_allclose(
            proj.S.numpy(), expected_S.numpy(), rtol=1e-12
        )

    def test_encode_decode_consistent_after_update(self, proj):
        Phi_new, Psi_new = _random_bases(N, R, seed=777)
        proj.update([Phi_new, Psi_new])

        c = torch.randn(R, dtype=torch.float64)
        q = proj.Phi @ c
        q_roundtrip = proj.decode(proj.encode(q))
        np.testing.assert_allclose(
            q_roundtrip.numpy(), q.numpy(), rtol=1e-10
        )


# ---------------------------------------------------------------------------
# vjp_encode tests
# ---------------------------------------------------------------------------

class TestVjpEncode:

    def test_finite_difference_Psi(self):
        """
        J(Psi) = v^T encode(q; Psi) = v^T Psi^T q.
        Check ⟨∂J/∂Psi, δPsi⟩ ≈ [J(Psi + ε δPsi) - J(Psi - ε δPsi)] / (2ε).
        """
        rng = np.random.default_rng(500)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        q = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        dPsi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)

        proj = LinearProjection(Phi, Psi)
        grads = proj.vjp_encode(q, v)
        dd_vjp = torch.sum(grads[1] * dPsi).item()

        eps = 1e-7
        proj_plus = LinearProjection(Phi, Psi + eps * dPsi)
        proj_minus = LinearProjection(Phi, Psi - eps * dPsi)
        J_plus = torch.dot(v, proj_plus.encode(q)).item()
        J_minus = torch.dot(v, proj_minus.encode(q)).item()
        dd_fd = (J_plus - J_minus) / (2 * eps)

        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)

    def test_finite_difference_Psi_batched(self):
        """
        Batched: J(Psi) = sum_m v_m^T Psi^T q_m.
        Batched inputs: q is (m, N), v is (m, r).
        """
        rng = np.random.default_rng(502)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        q = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        dPsi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)

        proj = LinearProjection(Phi, Psi)
        grads = proj.vjp_encode(q, v)
        assert grads[1].shape == (N, R)
        dd_vjp = torch.sum(grads[1] * dPsi).item()

        eps = 1e-7
        proj_plus = LinearProjection(Phi, Psi + eps * dPsi)
        proj_minus = LinearProjection(Phi, Psi - eps * dPsi)
        # J = sum_m v[m] . encode(q[m])
        J_plus = torch.sum(v * proj_plus.encode(q)).item()
        J_minus = torch.sum(v * proj_minus.encode(q)).item()
        dd_fd = (J_plus - J_minus) / (2 * eps)

        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# vjp_decode tests
# ---------------------------------------------------------------------------

class TestVjpDecode:

    def test_finite_difference_Phi(self):
        """
        J(Phi) = v^T decode(z; Phi, Psi) = v^T Phi S z.
        Check ⟨∂J/∂Phi, δPhi⟩ via central differences.
        """
        rng = np.random.default_rng(600)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
        dPhi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)

        proj = LinearProjection(Phi, Psi)
        grads = proj.vjp_decode(z, v)
        dd_vjp = torch.sum(grads[0] * dPhi).item()

        eps = 1e-7
        proj_plus = LinearProjection(Phi + eps * dPhi, Psi)
        proj_minus = LinearProjection(Phi - eps * dPhi, Psi)
        J_plus = torch.dot(v, proj_plus.decode(z)).item()
        J_minus = torch.dot(v, proj_minus.decode(z)).item()
        dd_fd = (J_plus - J_minus) / (2 * eps)

        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)

    def test_finite_difference_Psi(self):
        """
        J(Psi) = v^T decode(z; Phi, Psi) = v^T Phi (Psi^T Phi)^{-1} z.
        Check ⟨∂J/∂Psi, δPsi⟩ via central differences.
        """
        rng = np.random.default_rng(601)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
        dPsi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)

        proj = LinearProjection(Phi, Psi)
        grads = proj.vjp_decode(z, v)
        dd_vjp = torch.sum(grads[1] * dPsi).item()

        eps = 1e-7
        proj_plus = LinearProjection(Phi, Psi + eps * dPsi)
        proj_minus = LinearProjection(Phi, Psi - eps * dPsi)
        J_plus = torch.dot(v, proj_plus.decode(z)).item()
        J_minus = torch.dot(v, proj_minus.decode(z)).item()
        dd_fd = (J_plus - J_minus) / (2 * eps)

        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)

    def test_finite_difference_Phi_batched(self):
        """
        Batched: J(Phi) = sum_m v_m^T decode(z_m; Phi, Psi).
        Inputs: z is (m, r), v is (m, N).
        """
        rng = np.random.default_rng(602)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        dPhi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)

        proj = LinearProjection(Phi, Psi)
        grads = proj.vjp_decode(z, v)
        assert grads[0].shape == (N, R)
        dd_vjp = torch.sum(grads[0] * dPhi).item()

        eps = 1e-7
        proj_plus = LinearProjection(Phi + eps * dPhi, Psi)
        proj_minus = LinearProjection(Phi - eps * dPhi, Psi)
        J_plus = torch.sum(v * proj_plus.decode(z)).item()
        J_minus = torch.sum(v * proj_minus.decode(z)).item()
        dd_fd = (J_plus - J_minus) / (2 * eps)

        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)

    def test_finite_difference_Psi_batched(self):
        """
        Batched: J(Psi) = sum_m v_m^T decode(z_m; Phi, Psi).
        Inputs: z is (m, r), v is (m, N).
        """
        rng = np.random.default_rng(603)
        Phi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        Psi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)
        z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        dPsi = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)

        proj = LinearProjection(Phi, Psi)
        grads = proj.vjp_decode(z, v)
        assert grads[1].shape == (N, R)
        dd_vjp = torch.sum(grads[1] * dPsi).item()

        eps = 1e-7
        proj_plus = LinearProjection(Phi, Psi + eps * dPsi)
        proj_minus = LinearProjection(Phi, Psi - eps * dPsi)
        J_plus = torch.sum(v * proj_plus.decode(z)).item()
        J_minus = torch.sum(v * proj_minus.decode(z)).item()
        dd_fd = (J_plus - J_minus) / (2 * eps)

        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)
