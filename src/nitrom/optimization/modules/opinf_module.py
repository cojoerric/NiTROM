import torch
import torch.nn as nn

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.projections.projection import Projection
from nitrom.training_data import TrainingData

from .base import InferenceModule


class OpInfModule(InferenceModule):
    r"""
    Operator-inference module backed by :class:`PolynomialModel` or
    :class:`GasPolynomialModel`.

    Minimizes the weighted least-squares cost

    .. math::

        J = \sum_{i} w_i \lVert \dot{z}_i - f(t_i, z_i) \rVert^2
            + \lambda \sum_k \lVert A_k \rVert^2

    where :math:`z = \Phi^\top x`, :math:`\dot{z} = \Phi^\top \dot{x}`,
    and :math:`f` is evaluated by the underlying model.

    :param training_data: training data with ``X``, ``dX``, ``weights``, ``time``
    :type training_data: TrainingData
    :param latent_space_model: a pre-constructed latent-space dynamics model
        (e.g. :class:`PolynomialModel` or :class:`GasPolynomialModel`) whose
        parameters are optimized to fit the projected data.  Any fixed input
        operator (e.g. ``B = Phi^T B_fom``) and any initial guess should already
        be baked into the model (via its ``forcing_config`` / constructor).
    :type latent_space_model: PolynomialModel or GasPolynomialModel
    :param projection: a :class:`Projection` mapping the ambient state to the
        latent space; the training data are projected with
        :meth:`Projection.encode`
    :type projection: Projection
    :param reg: Tikhonov regularization weight
    :type reg: float
    """

    def __init__(
        self,
        training_data: TrainingData,
        latent_space_model: PolynomialModel | GasPolynomialModel,
        projection: Projection,
        reg: float = 0.0,
    ) -> None:
        super().__init__()

        self.reg = reg
        self.rom = latent_space_model
        self.projection = projection

        # Precompute projected data: Z, dZ of shape (ntraj, r, nt)
        self.Z = self._encode_trajectories(training_data.X)
        self.dZ = self._encode_trajectories(training_data.dX)

        ntraj, _, nt = self.Z.shape
        self.ntraj = ntraj
        self.nt = nt

        # Weight matrix
        W = (1 / training_data.weights.view(-1)).repeat_interleave(nt)
        self.W = torch.diag(W)

        # Store forcing callables and time grid
        self.forcing_fns = getattr(training_data, "forcing_fns", None)
        self.time = getattr(training_data, "time", None)

        # Register nn.Parameters mirroring the model's params
        for name in self.rom.param_names:
            self.register_parameter(
                name,
                nn.Parameter(getattr(self.rom, name).clone()),
            )

    def _encode_trajectories(self, A: torch.Tensor) -> torch.Tensor:
        r"""
        Encode a batch of ambient trajectories to the latent space.

        :meth:`Projection.encode` expects ``(N,)`` or ``(m, N)`` inputs, so the
        time axis is flattened into the batch dimension before encoding and
        restored afterwards.

        :param A: ambient trajectories of shape ``(ntraj, N, nt)``
        :type A: torch.Tensor
        :returns: latent trajectories of shape ``(ntraj, r, nt)``
        :rtype: torch.Tensor
        """
        ntraj, N, nt = A.shape
        A_flat = A.permute(0, 2, 1).reshape(-1, N)  # (ntraj * nt, N)
        Z_flat = self.projection.encode(A_flat)  # (ntraj * nt, r)
        r = Z_flat.shape[-1]
        return Z_flat.reshape(ntraj, nt, r).permute(0, 2, 1)  # (ntraj, r, nt)

    def _sync_to_rom(self) -> None:
        """Push current nn.Parameters into the underlying ROM."""
        tensors = [getattr(self, name) for name in self.rom.param_names]
        self.rom.update_params(tensors)

    def _evaluate_rhs_all(self) -> torch.Tensor:
        r"""
        Evaluate the ROM RHS at every ``(traj, time)`` pair.

        When no forcing is present, all snapshots are batched in a single
        call.  When forcing callables exist, the evaluation loops over
        time snapshots so each call receives the correct ``t`` and
        ``external_forcing``.

        :returns: ``fZ`` of shape ``(ntraj, r, nt)``
        :rtype: torch.Tensor
        """
        r = self.rom.state_dimension

        if self.forcing_fns is None or len(self.forcing_fns) == 0:
            Z_flat = self.Z.permute(0, 2, 1).reshape(-1, r)
            fZ = self.rom.evaluate_rhs(0.0, Z_flat)
            return fZ.reshape(self.ntraj, self.nt, r).permute(0, 2, 1)

        # Loop over time snapshots to pass per-trajectory forcing
        fZ = torch.zeros_like(self.dZ)  # (ntraj, r, nt)
        for j in range(self.nt):
            t_j = self.time[j]
            z_j = self.Z[:, :, j]  # (ntraj, r)
            fZ[:, :, j] = self.rom.evaluate_rhs(
                t_j, z_j, external_forcing=self.forcing_fns,
            )
        return fZ

    def forward(self) -> torch.Tensor:
        r"""
        Evaluate the weighted least-squares cost.

        :returns: scalar loss
        :rtype: torch.Tensor
        """
        self._sync_to_rom()

        fZ = self._evaluate_rhs_all()

        R = self.dZ - fZ
        R_flat = R.permute(1, 0, 2).reshape(self.rom.state_dimension, -1)
        cost = ((R_flat @ self.W) * R_flat).sum()

        # Regularization
        for tensor in self.rom.get_params():
            cost = cost + self.reg * torch.norm(tensor) ** 2

        return cost

    def gradient(self) -> list[torch.Tensor]:
        r"""
        Compute analytic gradients of the cost w.r.t. the trainable
        parameters using the model's VJP.

        :returns: list of gradient tensors, one per parameter
        :rtype: list[torch.Tensor]
        """
        self._sync_to_rom()
        r = self.rom.state_dimension

        # Compute residual
        fZ = self._evaluate_rhs_all()

        R = self.dZ - fZ
        R_flat = R.permute(1, 0, 2).reshape(r, -1)
        RW = (R_flat @ self.W).reshape(r, self.ntraj, self.nt).permute(1, 0, 2)

        # VJP: adjoint seed v = -2 * R * W
        if self.forcing_fns is None or len(self.forcing_fns) == 0:
            Z_flat = self.Z.permute(0, 2, 1).reshape(-1, r)
            v = -2.0 * RW.permute(0, 2, 1).reshape(-1, r)
            grads = self.rom.vjp_evaluate_rhs(Z_flat, v)
        else:
            # Accumulate VJP over time snapshots
            grads = None
            for j in range(self.nt):
                t_j = self.time[j]
                z_j = self.Z[:, :, j]           # (ntraj, r)
                v_j = -2.0 * RW[:, :, j]        # (ntraj, r)
                grads_j = self.rom.vjp_evaluate_rhs(
                    z_j, v_j,
                    external_forcing=self.forcing_fns, t=t_j,
                )
                if grads is None:
                    grads = grads_j
                else:
                    for i in range(len(grads)):
                        grads[i] = grads[i] + grads_j[i]

        # Add regularization gradients
        params = self.rom.get_params()
        for i in range(len(grads)):
            grads[i] = grads[i] + 2.0 * self.reg * params[i]

        # Zero the gradient of any non-learnable parameter (base class).
        return self._apply_learnability(grads)
