import torch
import torch.nn as nn

from nitrom.projections.polynomial_projection import PolynomialProjection
from nitrom.training_data import TrainingData

from .base import InferenceModule


class PolyManifoldInfModule(InferenceModule):
    r"""
    Polynomial manifold inference module backed by
    :class:`PolynomialProjection`.

    Minimizes the reconstruction cost

    .. math::

        J = \sum_{i} w_i \lVert x_i - \text{decode}(z_i) \rVert^2
            + \lambda \sum_k \lVert A_k \rVert^2

    where :math:`z_i = \Psi^\top x_i` and

    .. math::

        \text{decode}(z) = \Phi S\, z
            + \mathbb{P}\sum_k A_k\, z^{\otimes k},
        \quad
        \mathbb{P} = I - \Phi S\, \Psi^\top.

    The bases :math:`\Phi` and :math:`\Psi` are **fixed**; only the
    nonlinear tensors :math:`A_k` are learned.

    :param training_data: training data with ``X`` of shape
        ``(ntraj, N, nt)`` and ``weights`` of shape ``(ntraj,)``
    :type training_data: TrainingData
    :param nonlin_poly_comp: nonlinear polynomial degrees in ascending
        order, e.g. ``[2, 3]``
    :type nonlin_poly_comp: list[int]
    :param Phi: trial basis of shape ``(N, r)``
    :type Phi: torch.Tensor
    :param Psi: test basis of shape ``(N, r)``.  If ``None``, defaults
        to ``Phi`` (orthogonal projection).
    :type Psi: torch.Tensor or None
    :param reg: Tikhonov regularization weight on the ``A_k`` tensors
    :type reg: float
    :param initial_guess: optional list of initial ``A_k`` tensors,
        one per entry in ``nonlin_poly_comp``.  If ``None``, tensors are
        initialized to zero.
    :type initial_guess: list[torch.Tensor] or None
    """

    def __init__(
        self,
        training_data: TrainingData,
        nonlin_poly_comp: list[int],
        Phi: torch.Tensor,
        Psi: torch.Tensor | None = None,
        reg: float = 0.0,
        initial_guess: list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()

        if Psi is None:
            Psi = Phi

        N, r = Phi.shape
        dev = Phi.device
        dtype = Phi.dtype

        self.nonlin_poly_comp = nonlin_poly_comp
        self.reg = reg

        # Store full-space data: X of shape (ntraj, N, nt)
        self.X = training_data.X

        ntraj, _, nt = self.X.shape
        self.ntraj = ntraj
        self.nt = nt

        # Weight matrix
        W = (1.0 / training_data.weights).repeat_interleave(nt)
        self.W = torch.diag(W)

        # Precompute encoded data: Z of shape (ntraj, r, nt)
        self.Z = torch.einsum("ij,kil->kjl", Psi, self.X)

        # Build initial A_k tensors
        if initial_guess is not None:
            if len(initial_guess) != len(nonlin_poly_comp):
                raise ValueError(
                    f"initial_guess has {len(initial_guess)} tensors, "
                    f"expected {len(nonlin_poly_comp)}"
                )
            proj_tensors = [Phi, Psi] + [
                t.to(device=dev, dtype=dtype) for t in initial_guess
            ]
        else:
            proj_tensors = [Phi, Psi] + [
                torch.zeros((N,) + (r,) * k, device=dev, dtype=dtype)
                for k in nonlin_poly_comp
            ]

        # Create the underlying PolynomialProjection
        self.proj = PolynomialProjection(nonlin_poly_comp, proj_tensors)

        # Register only the A_k tensors as nn.Parameters (Phi, Psi are fixed)
        for k in nonlin_poly_comp:
            name = f"A{k}"
            self.register_parameter(
                name,
                nn.Parameter(getattr(self.proj, name).clone()),
            )

    @property
    def _trainable_names(self) -> list[str]:
        """Names of the trainable A_k parameters."""
        return [f"A{k}" for k in self.nonlin_poly_comp]

    def _sync_to_proj(self) -> None:
        """Push current nn.Parameters into the underlying projection."""
        params = [self.proj.Phi, self.proj.Psi] + [
            getattr(self, name) for name in self._trainable_names
        ]
        self.proj.update(params)

    def forward(self) -> torch.Tensor:
        r"""
        Evaluate the reconstruction cost.

        .. math::

            J = \sum_{i} w_i \lVert x_i - \text{decode}(z_i) \rVert^2
                + \lambda \sum_k \lVert A_k \rVert^2

        :returns: scalar loss
        :rtype: torch.Tensor
        """
        self._sync_to_proj()

        # Z_flat: (ntraj*nt, r),  X_flat: (ntraj*nt, N)
        Z_flat = self.Z.permute(0, 2, 1).reshape(-1, self.proj.latent_space_dimension)
        X_flat = self.X.permute(0, 2, 1).reshape(-1, self.proj.ambient_space_dimension)

        X_hat = self.proj.decode(Z_flat)  # (ntraj*nt, N)

        R = X_flat - X_hat  # (ntraj*nt, N)
        # Reshape to (N, ntraj*nt) for weighted norm
        R_flat = R.T
        cost = ((R_flat @ self.W) * R_flat).sum()

        # Regularization on A_k
        for name in self._trainable_names:
            cost = cost + self.reg * torch.norm(getattr(self.proj, name)) ** 2

        return cost

    def gradient(self) -> list[torch.Tensor]:
        r"""
        Compute analytic gradients of the cost w.r.t. the trainable
        ``A_k`` tensors using the projection's VJP.

        :returns: list of gradient tensors, one per ``A_k``
        :rtype: list[torch.Tensor]
        """
        self._sync_to_proj()

        r = self.proj.latent_space_dimension
        N = self.proj.ambient_space_dimension

        Z_flat = self.Z.permute(0, 2, 1).reshape(-1, r)
        X_flat = self.X.permute(0, 2, 1).reshape(-1, N)

        X_hat = self.proj.decode(Z_flat)

        R = X_flat - X_hat
        R_flat = R.T  # (N, ntraj*nt)
        RW = (R_flat @ self.W).T  # (ntraj*nt, N)

        # Adjoint seed: v = -2 * R * W
        v = -2.0 * RW

        # Full VJP returns (grad_Phi, grad_Psi, grad_A_k1, ...)
        all_grads = self.proj.vjp_decode(Z_flat, v)

        # Extract only the A_k gradients (skip grad_Phi, grad_Psi)
        grads = list(all_grads[2:])

        # Add regularization
        for i, name in enumerate(self._trainable_names):
            grads[i] = grads[i] + 2.0 * self.reg * getattr(self.proj, name)

        return grads
