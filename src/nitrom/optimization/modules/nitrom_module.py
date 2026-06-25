import numpy as np
import torch
import torch.nn as nn

from nitrom.roms.param_registry import ParamRegistry
from nitrom.training_data import TrainingData

from ...time_steppers.time_stepper import solve_ivp
from ...utils import interp_quadratic
from .base import InferenceModule


class NitromModule(InferenceModule):
    r"""
    NiTROM training module backed by a :class:`ParamRegistry`.

    Unlike :class:`OpInfModule`, which owns a single latent-space model,
    this module receives a :class:`ParamRegistry` that already holds both
    the latent-space :class:`Model` and the :class:`Projection`.  The
    module mirrors the registry's parameters as trainable
    ``nn.Parameter`` s (shared parameters appear exactly once) and pushes
    them back into the underlying model and projection through
    :meth:`ParamRegistry.scatter`, which keeps shared parameters tied by
    construction.

    The cost (:meth:`forward`) and its analytic :meth:`gradient` are left
    to be implemented.

    :param training_data: training data (trajectories ``X``, time grid,
        forcing callables, weights)
    :type training_data: TrainingData
    :param registry: registry of the ROM parameters spanning the
        latent-space model and the projection
    :type registry: ParamRegistry
    :param fom: optional full-order model, e.g. for output evaluation
    :type fom: object or None
    :param reg: Tikhonov regularization weight
    :type reg: float
    """

    def __init__(
        self,
        training_data: TrainingData,
        registry: ParamRegistry,
        fom=None,
        reg: float = 0.0,
        n_substeps: int = 100,
        time_stepper: str = "rk4",
        n_leggauss: int = 5,
    ) -> None:
        super().__init__()

        if time_stepper not in ("rk2", "rk4"):
            raise ValueError(
                f"time_stepper must be 'rk2' or 'rk4', got {time_stepper!r}."
            )

        self.registry = registry
        self.training_data = training_data
        self.fom = fom
        self.reg = reg
        # Number of sub-steps per snapshot interval, shared by the forward
        # solve and the backward adjoint so the two stay consistent.
        self.n_substeps = n_substeps
        # Runge-Kutta scheme used for both the forward and adjoint solves.
        self.time_stepper = time_stepper
        # Number of Gauss-Legendre nodes for the parameter-gradient integrals.
        self.n_leggauss = n_leggauss

        # Convenience handles into the registry's components
        self.model = registry.model
        self.projection = registry.projection

        # Gauss-Legendre nodes / weights on [-1, 1] for quadrature of the
        # adjoint parameter-gradient integrals.
        nodes, weights = np.polynomial.legendre.leggauss(n_leggauss)
        self._gl_nodes = torch.tensor(
            nodes, device=self.model.device, dtype=self.model.dtype
        )
        self._gl_weights = torch.tensor(
            weights, device=self.model.device, dtype=self.model.dtype
        )

        # Commonly used training data (mirrors OpInfModule conventions)
        self.forcing_fns = getattr(training_data, "forcing_fns", None)
        self.time = getattr(training_data, "time", None)
        self.weights = getattr(training_data, "weights", None)

        # Register one nn.Parameter per registered parameter, in registry
        # order.  Shared parameters appear once.
        for name in self.registry.names:
            self.register_parameter(
                name,
                nn.Parameter(self.registry.value(name).clone()),
            )

    @property
    def param_names(self) -> list[str]:
        """Names of the trainable parameters, in registry order."""
        return self.registry.names

    def parameter_list(self) -> list[torch.Tensor]:
        """Current trainable parameter tensors, in registry order."""
        return [getattr(self, name) for name in self.registry.names]

    def _sync_to_registry(self) -> None:
        """
        Push the current ``nn.Parameter`` values into the model and
        projection via :meth:`ParamRegistry.scatter`, keeping shared
        parameters identical across both components.
        """
        self.registry.scatter(self.parameter_list())

    def _decode_trajectories(self, Z: torch.Tensor) -> torch.Tensor:
        r"""
        Decode a batch of latent trajectories to the ambient space.

        :meth:`Projection.decode` expects ``(r,)`` or ``(m, r)`` inputs, so
        the time axis is flattened into the batch dimension before decoding
        and restored afterwards.

        :param Z: latent trajectories of shape ``(ntraj, r, nt)``
        :type Z: torch.Tensor
        :returns: ambient trajectories of shape ``(ntraj, N, nt)``
        :rtype: torch.Tensor
        """
        ntraj, r, nt = Z.shape
        Z_flat = Z.permute(0, 2, 1).reshape(-1, r)  # (ntraj * nt, r)
        X_flat = self.projection.decode(Z_flat)  # (ntraj * nt, N)
        N = X_flat.shape[-1]
        return X_flat.reshape(ntraj, nt, N).permute(0, 2, 1)  # (ntraj, N, nt)

    def forward(self) -> torch.Tensor:
        r"""
        Evaluate the NiTROM cost, defined as

        .. math::

            J = \sum_{j=0}^{ntraj-1} \alpha_j^{-1}\sum_{i=0}^{nt-1}
                \lVert y^{(j)}(t_i) - \hat{y}^{(j)}(t_i) \rVert^2

        where :math:`y^{(j)}(t_i)` is the output of the full-order model
        at time :math:`t_i` for trajectory :math:`j`,
        and :math:`\hat{y}^{(j)}(t_i) = h\left(D z^{(j)}(t_i)\right)` is the output of the reduced-order model.

        The per-trajectory weights :math:`\alpha_j` are read from
        ``training_data.weights`` (mirroring :class:`OpInfModule`), the time
        grid from ``training_data.time``, the initial conditions / true states
        from ``training_data.X`` (shape ``(ntraj, N, nt)``), and the forcing
        callables from ``training_data.forcing_fns``.  The output map
        :math:`h` is supplied by the full-order model as ``fom.compute_output``.

        :returns: scalar loss
        :rtype: torch.Tensor
        """
        # Push current parameters into the model and projection (keeps any
        # shared parameters tied across the two components).
        self._sync_to_registry()

        # Encode the initial conditions to the latent space.
        z0 = self.projection.encode(self.training_data.X[:, :, 0])  # (ntraj, r)

        # Integrate the latent dynamics over the trajectory time grid.
        # solve_ivp forwards **kwargs to the RHS, so the forcing callables
        # reach evaluate_rhs as the ``external_forcing`` keyword.
        dt = (self.time[1] - self.time[0]) / self.n_substeps
        Z = solve_ivp(
            self.model.evaluate_rhs,
            z0,
            self.time[0],
            self.time[-1],
            dt,
            self.time,
            self.time_stepper,
            external_forcing=self.forcing_fns or None,
        )  # (ntraj, r, nt)

        # Weighted sum-of-squares output mismatch.
        e = self.fom.compute_output(self.training_data.X) - self.fom.compute_output(
            self._decode_trajectories(Z)
        )
        J = torch.sum((e * e).sum(dim=(1, 2)) / self.weights)
        return J

    def _vjp_rhs(
        self, z: torch.Tensor, lam: torch.Tensor, t: float
    ) -> list[torch.Tensor]:
        """
        VJP of the latent RHS w.r.t. the model parameters, forwarding the
        forcing only when the model carries a learnable input operator.
        """
        if getattr(self.model, "forcing_exists", False) and self.forcing_fns:
            return self.model.vjp_evaluate_rhs(
                z, lam, external_forcing=self.forcing_fns, t=t
            )
        return self.model.vjp_evaluate_rhs(z, lam)

    def gradient(self) -> list[torch.Tensor]:
        r"""
        Compute the analytic gradient of the cost w.r.t. the trainable
        parameters, in registry order, by the adjoint method.

        The cost depends on the parameters through three paths, each handled
        with the corresponding hand-derived VJP:

        * **latent dynamics** (model parameters) -- the adjoint
          :math:`\lambda(t)` is propagated backward with
          :meth:`Model.evaluate_adjoint_rhs` and the gradient accumulated as
          :math:`\int (\partial f/\partial\mu)^\top \lambda\,dt` via
          :meth:`Model.vjp_evaluate_rhs`;
        * **encoder** (initial condition :math:`z(0)=\Psi^\top x_0`) -- via
          :meth:`Projection.vjp_encode` seeded with :math:`\lambda(0)`;
        * **decoder** (:math:`\hat y_i = h(D z(t_i))`) -- via
          :meth:`Projection.vjp_decode` seeded at every measurement time.

        At each measurement time the output residual is mapped into the
        latent space (the adjoint source) through the linear part of the
        decoder, :math:`S^\top \Phi^\top`, matching the construction in
        ``nitrom_cost_and_grad.py``.  Contributions to a shared parameter
        (one that appears in both the model and the projection) are summed.

        .. note:: This is a *continuous* adjoint.  The adjoint is integrated
            backward with the :attr:`time_stepper` scheme onto
            :attr:`n_leggauss` Gauss-Legendre nodes per interval, and the
            parameter-gradient integrals are evaluated by Gauss-Legendre
            quadrature; it matches a finite-difference of :meth:`forward` to
            discretization accuracy.  The output map is assumed linear,
            ``fom.compute_output_derivative`` returning the constant operator
            of shape ``(no, N)``.

        :returns: list of gradient tensors, one per registered parameter
        :rtype: list[torch.Tensor]
        """
        with torch.no_grad():
            self._sync_to_registry()

            X = self.training_data.X  # (ntraj, N, nt)
            time = self.time  # (nt,)
            ntraj, _, nt = X.shape
            r = self.model.state_dimension
            ef = self.forcing_fns or None
            w = (1.0 / self.weights).reshape(-1, 1, 1)  # (ntraj, 1, 1)

            # --- forward solve at the measurement times --------------------
            z0 = self.projection.encode(X[:, :, 0])  # (ntraj, r)
            dt = (time[1] - time[0]) / self.n_substeps
            Z = solve_ivp(
                self.model.evaluate_rhs,
                z0,
                time[0],
                time[-1],
                dt,
                time,
                self.time_stepper,
                external_forcing=ef,
            )  # (ntraj, r, nt)

            # --- output residual and adjoint sources at each snapshot ------
            Xhat = self._decode_trajectories(Z)  # (ntraj, N, nt)
            e = self.fom.compute_output(X) - self.fom.compute_output(Xhat)
            C = self.fom.compute_output_derivative(Xhat)  # (no, N), constant
            # Weighted full-space output seed v_i = -2 alpha_j^{-1} C^T e_i.
            cw = -2.0 * w * torch.einsum("on,bot->bnt", C, e)  # (ntraj, N, nt)
            N = cw.shape[1]

            # Flatten the (trajectory, snapshot) axes into a single batch so the
            # per-snapshot projection VJPs are evaluated in one shot.
            Z_flat = Z.permute(0, 2, 1).reshape(-1, r)  # (ntraj * nt, r)
            cw_flat = cw.permute(0, 2, 1).reshape(-1, N)  # (ntraj * nt, N)

            # Latent adjoint source: decoder Jacobian-transpose D_z^T applied to
            # the weighted output seed (per-sample, no batch reduction).
            # Delegated to the projection so the adjoint is correct for any
            # (possibly nonlinear) decoder.
            src = (
                self.projection.vjp_decode_state(Z_flat, cw_flat)
                .reshape(ntraj, nt, r)
                .permute(0, 2, 1)
            )  # (ntraj, r, nt)

            # Decoder parameter gradient: vjp_decode sums over its batch, so a
            # single call over the flattened batch yields the sum over all
            # (trajectory, snapshot) pairs.
            proj_grads = list(self.projection.vjp_decode(Z_flat, cw_flat))

            # --- backward adjoint sweep ------------------------------------
            model_grads = [torch.zeros_like(p) for p in self.model.get_params()]
            lam = torch.zeros((ntraj, r), device=X.device, dtype=X.dtype)
            xi, wq = self._gl_nodes, self._gl_weights

            for k in range(nt - 1, 0, -1):
                # Inject the measurement source at snapshot k.
                lam = lam + src[:, :, k]

                # Re-integrate the base flow over [t_{k-1}, t_k].
                t0i, tfi = float(time[k - 1]), float(time[k])
                delta = tfi - t0i
                a = 0.5 * delta
                sub_t = torch.linspace(
                    t0i, tfi, self.n_substeps + 1, device=X.device, dtype=X.dtype
                )
                Zint = solve_ivp(
                    self.model.evaluate_rhs,
                    Z[:, :, k - 1],
                    t0i,
                    tfi,
                    delta / self.n_substeps,
                    sub_t,
                    self.time_stepper,
                    external_forcing=ef,
                )  # (ntraj, r, n_substeps + 1)

                # Adjoint in reversed time tau in [0, delta] (physical time
                # t = tfi - tau): d(lam)/d(tau) = J_f(Z(t))^T lam.  The base
                # flow at the integrator's stage times is interpolated from Zint.
                def adj_rhs(tau, lam_, _Zint=Zint, _sub_t=sub_t, _tfi=tfi):
                    phys_t = _tfi - tau
                    tq = torch.atleast_1d(
                        torch.as_tensor(phys_t, device=X.device, dtype=X.dtype)
                    )
                    Z_t = interp_quadratic(tq, _sub_t, _Zint)[..., 0]  # (ntraj, r)
                    return self.model.evaluate_adjoint_rhs(float(phys_t), lam_, Z_t)

                # Integrate the adjoint directly onto the Gauss-Legendre nodes
                # (reversed-time), plus the interval end for the carry-forward.
                tau_nodes = a * (1.0 - xi)  # (n_leggauss,)
                order = torch.argsort(tau_nodes)
                tau_eval = torch.cat(
                    [
                        tau_nodes[order],
                        torch.tensor([delta], device=X.device, dtype=X.dtype),
                    ]
                )
                Lam_sol = solve_ivp(
                    adj_rhs,
                    lam,
                    0.0,
                    delta,
                    delta / self.n_substeps,
                    tau_eval,
                    self.time_stepper,
                )  # (ntraj, r, n_leggauss + 1)

                # Adjoint at the GL nodes (undo the sort); carry lambda(t_{k-1}).
                Lam_nodes = torch.empty(
                    (ntraj, r, self.n_leggauss), device=X.device, dtype=X.dtype
                )
                Lam_nodes[..., order] = Lam_sol[..., : self.n_leggauss]
                lam = Lam_sol[..., -1]

                # Base flow at the physical GL nodes.
                t_gl = 0.5 * (tfi + t0i) + a * xi  # (n_leggauss,)
                Z_nodes = interp_quadratic(t_gl, sub_t, Zint)  # (ntraj, r, n_leggauss)

                # Gauss-Legendre quadrature of the model-parameter integral:
                # int_{t_{k-1}}^{t_k} (df/dmu)^T lam dt = a * sum_i w_i vjp(...).
                # Fold the quadrature weight a * w_i into the adjoint seed (the
                # VJP is linear in it) so the node sum is a batch reduction.
                Lam_w = Lam_nodes * (a * wq).reshape(1, 1, -1)  # (ntraj, r, n_leggauss)
                if getattr(self.model, "forcing_exists", False) and self.forcing_fns:
                    # Forcing makes the VJP depend on the per-node time, so the
                    # input-operator gradient must be accumulated node by node.
                    for i in range(self.n_leggauss):
                        g = self._vjp_rhs(
                            Z_nodes[..., i], Lam_w[..., i], float(t_gl[i])
                        )
                        for idx in range(len(model_grads)):
                            model_grads[idx] = model_grads[idx] + g[idx]
                else:
                    # Flatten (trajectory, node) and reduce in one VJP call.
                    Zf = Z_nodes.permute(0, 2, 1).reshape(-1, r)
                    Lf = Lam_w.permute(0, 2, 1).reshape(-1, r)
                    for idx, g in enumerate(self.model.vjp_evaluate_rhs(Zf, Lf)):
                        model_grads[idx] = model_grads[idx] + g

            # Measurement at t_0, then encoder gradient seeded with lambda(0).
            lam = lam + src[:, :, 0]
            for k, g in enumerate(self.projection.vjp_encode(X[:, :, 0], lam)):
                proj_grads[k] = proj_grads[k] + g

            # --- assemble in registry order, summing shared contributions --
            grad_by_name: dict[str, torch.Tensor] = {}
            for name, g in zip(self.projection.param_names, proj_grads, strict=True):
                grad_by_name[name] = g
            for name, g in zip(self.model.param_names, model_grads, strict=True):
                grad_by_name[name] = (
                    grad_by_name[name] + g if name in grad_by_name else g
                )
            return [grad_by_name[name] for name in self.registry.names]
