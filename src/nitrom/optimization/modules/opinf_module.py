import torch
import torch.nn as nn

from nitrom.latent_space_models.gas_polynomial_model import GasPolynomialModel
from nitrom.latent_space_models.polynomial_model import PolynomialModel
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
    :param poly_comp: polynomial degrees, e.g. ``[1, 2]``
    :type poly_comp: list[int]
    :param Phi: trial basis of shape ``(N, r)``
    :type Phi: torch.Tensor
    :param reg: Tikhonov regularization weight
    :type reg: float
    :param initial_guess: optional list of tensors to initialize the
        model parameters.  For standard mode, must match
        ``len(poly_comp)``; for GAS mode, must match the number of
        GAS parameters.
    :type initial_guess: list[torch.Tensor] or None
    :param gas_flag: if ``True``, use :class:`GasPolynomialModel`
    :type gas_flag: bool
    :param forcing_config: optional dict with keys ``"forcing_exists"``
        (bool) and ``"m"`` (int).  Passed through to the underlying model.
    :type forcing_config: dict or None
    """

    def __init__(
        self,
        training_data: TrainingData,
        poly_comp: list[int],
        Phi: torch.Tensor,
        reg: float = 0.0,
        initial_guess: list[torch.Tensor] | None = None,
        gas_flag: bool = False,
        forcing_config: dict | None = None,
    ) -> None:
        super().__init__()

        self.poly_comp = poly_comp
        self.reg = reg
        self.gas_flag = gas_flag

        r = Phi.shape[-1]
        dev = Phi.device
        dtype = Phi.dtype

        # If a full-order input matrix ``B_fom`` is supplied, project it onto
        # the basis to obtain a fixed (non-trainable) reduced input operator
        # B = Phi^T B_fom.
        if forcing_config is not None and forcing_config.get("B_fom") is not None:
            B_fom = forcing_config["B_fom"].to(device=dev, dtype=dtype)
            forcing_config = {
                **forcing_config,
                "B": Phi.T @ B_fom,
                "m": B_fom.shape[1],
            }

        self.forcing_config = forcing_config
        self.forcing_exists = forcing_config is not None and forcing_config.get(
            "forcing_exists", False
        )
        self.m = forcing_config.get("m") if forcing_config is not None else None

        # Precompute projected data: Z, dZ of shape (ntraj, r, nt)
        self.Z = torch.einsum("ij,kil->kjl", Phi, training_data.X)
        self.dZ = torch.einsum("ij,kil->kjl", Phi, training_data.dX)

        ntraj, _, nt = self.Z.shape
        self.ntraj = ntraj
        self.nt = nt

        # Weight matrix
        W = (1 / training_data.weights).repeat_interleave(nt)
        self.W = torch.diag(W)

        # Create the underlying model
        if gas_flag:
            self.rom = GasPolynomialModel(
                r, poly_comp, device=dev, dtype=dtype,
                gas_params=initial_guess,
                forcing_config=forcing_config,
            )
        else:
            self.rom = PolynomialModel(
                r, poly_comp, device=dev, dtype=dtype,
                tensors=initial_guess,
                forcing_config=forcing_config,
            )

        # Store forcing callables and time grid
        self.forcing_fns = getattr(training_data, "forcing_fns", None)
        self.time = getattr(training_data, "time", None)

        # Register nn.Parameters mirroring the model's params
        for name in self.rom.param_names:
            self.register_parameter(
                name,
                nn.Parameter(getattr(self.rom, name).clone()),
            )

        # Per-parameter learnability (all trainable by default).  Non-learnable
        # parameters have their gradient zeroed in :meth:`gradient`.
        self.is_learnable = dict.fromkeys(self.rom.param_names, True)

    def set_unlearnable(self, *names: str) -> None:
        """
        Mark one or more parameters as non-learnable.

        All parameters are learnable by default.  A non-learnable parameter
        keeps its current value during training: its gradient is zeroed both
        in the model VJP and in the regularization term, so the optimizer
        never updates it.  Typical use is to freeze the input operator,
        ``model.set_unlearnable("B")``.

        :param names: parameter names, each in :attr:`rom.param_names`
        :type names: str
        :raises KeyError: if a name is not a parameter of the underlying ROM
        """
        self._set_learnable(names, False)

    def set_learnable(self, *names: str) -> None:
        """
        Mark one or more parameters as learnable (the default state).

        Reverses :meth:`set_unlearnable`, so the optimizer updates these
        parameters again.

        :param names: parameter names, each in :attr:`rom.param_names`
        :type names: str
        :raises KeyError: if a name is not a parameter of the underlying ROM
        """
        self._set_learnable(names, True)

    def _set_learnable(self, names: tuple[str, ...], value: bool) -> None:
        for name in names:
            if name not in self.is_learnable:
                raise KeyError(
                    f"Unknown parameter '{name}'; expected one of "
                    f"{list(self.is_learnable)}."
                )
            self.is_learnable[name] = value

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

        # Zero the gradient of any non-learnable parameter so the optimizer
        # leaves it fixed at its current value.
        for i, name in enumerate(self.rom.param_names):
            if not self.is_learnable[name]:
                grads[i] = torch.zeros_like(grads[i])

        return grads
