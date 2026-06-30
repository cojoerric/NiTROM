import torch

from .model import Model
from .polynomial_model import PolynomialModel


class GasPolynomialModel(Model):
    r"""
    GAS-constrained polynomial ROM.

    Holds free GAS parameters (``K``, ``R``, ``Q``, ``S``) and assembles
    physical operator tensors :math:`A` and :math:`H` via
    structure-preserving maps:

    .. math::

        A = \bigl((K - K^\top) - R^{-1}\,R^{-\top}\bigr)\,\tilde{Q},
        \qquad
        H_{ijk} = (S_{ilk} - S_{lik})\,\tilde{Q}_{lj},

    where :math:`\tilde{Q} = Q^{-1} Q^{-\top}`.

    All RHS, adjoint, and VJP evaluations are delegated to an internal
    :class:`PolynomialModel`.

    :param r: reduced state dimension
    :type r: int
    :param poly_comp: polynomial degrees, e.g. ``[1, 2]``
    :type poly_comp: list[int]
    :param device: device for tensor allocation
    :type device: torch.device or str
    :param dtype: data type for tensors
    :type dtype: torch.dtype
    :param instability_threshold: norm threshold for blow-up guard
    :type instability_threshold: float
    :param gas_params: optional list of initial GAS parameter tensors
        (subset depending on ``poly_comp``).
        If ``None``, parameters are initialized randomly.
    :type gas_params: list[torch.Tensor] or None
    :param forcing_config: optional dict with keys ``"forcing_exists"``
        (bool) and ``"m"`` (int).  See :class:`PolynomialModel`.
    :type forcing_config: dict or None
    """

    def __init__(
        self,
        r: int,
        poly_comp: list[int],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        instability_threshold: float = 1e6,
        gas_params: list[torch.Tensor] | None = None,
        forcing_config: dict | None = None,
    ):
        # Determine GAS parameter names and shapes
        param_names: list[str] = []
        gas_shapes: list[tuple[int, ...]] = []
        if 1 in poly_comp:
            param_names.extend(["K", "R"])
            gas_shapes.extend([(r, r), (r, r)])
        if 2 in poly_comp:
            param_names.extend(["Q", "S"])
            gas_shapes.extend([(r, r), (r, r, r)])

        # Track forcing.  An optional fixed input operator may be supplied via
        # ``forcing_config["B"]``; B stays a parameter but is flagged
        # non-learnable so that its gradient is zeroed.
        forcing_exists = forcing_config is not None and forcing_config.get(
            "forcing_exists", False
        )
        B_fixed = forcing_config.get("B") if forcing_config else None
        if forcing_exists:
            param_names.append("B")

        super().__init__(r, param_names, device, dtype)
        self.poly_comp = poly_comp
        self.forcing_exists = forcing_exists

        # Set GAS parameters as attributes
        if gas_params is not None:
            for name, tensor in zip(param_names, gas_params):
                setattr(self, name, tensor.to(device=self.device, dtype=self.dtype))
        else:
            # Initialize the GAS parameters randomly (K, R, Q, S; B handled below)
            gas_param_names = [name for name in param_names if name != "B"]
            for name, shape in zip(gas_param_names, gas_shapes, strict=True):
                setattr(
                    self, name, torch.randn(shape, device=self.device, dtype=self.dtype)
                )
            # Initialize B (fixed value if supplied, else zeros)
            if forcing_exists:
                m = forcing_config["m"]
                self.B = (
                    B_fixed.to(device=self.device, dtype=self.dtype)
                    if B_fixed is not None
                    else torch.zeros((r, m), device=self.device, dtype=self.dtype)
                )

        # A supplied fixed B always overrides any value from gas_params.
        if forcing_exists and B_fixed is not None:
            self.B = B_fixed.to(device=self.device, dtype=self.dtype)

        # Assemble physical tensors and create the inner PolynomialModel
        tensors = self.assemble_gas_tensors()
        self.model = PolynomialModel(
            r,
            poly_comp,
            device=device,
            dtype=dtype,
            instability_threshold=instability_threshold,
            tensors=tensors,
            forcing_config=forcing_config,
        )

    def get_params(self) -> list[torch.Tensor]:
        """Return the current GAS parameter tensors as a list."""
        return [getattr(self, name) for name in self.param_names]

    def assemble_gas_tensors(self) -> list[torch.Tensor]:
        r"""
        Build physical operator tensors from the current GAS parameters.

        :returns: list of tensors ``[A, H, ..., B]`` matching the inner
            :class:`PolynomialModel` param order.  ``B`` is appended
            only when forcing is present.
        :rtype: list[torch.Tensor]
        """
        tensors = [None] * len(self.poly_comp)

        Qinv = torch.linalg.inv(self.Q)
        Qtil = Qinv @ Qinv.T

        if 1 in self.poly_comp:
            idx = self.poly_comp.index(1)
            Rinv = torch.linalg.inv(self.R)
            tensors[idx] = ((self.K - self.K.T) - Rinv @ Rinv.T) @ Qtil

        if 2 in self.poly_comp:
            idx = self.poly_comp.index(2)
            tensors[idx] = torch.einsum("ilk,lj->ijk", self.S, Qtil) - torch.einsum(
                "lik,lj->ijk", self.S, Qtil
            )

        if self.forcing_exists:
            tensors.append(self.B)

        return tensors

    def update_params(self, params: list[torch.Tensor]) -> None:
        r"""
        Update the GAS parameters (and B if present), reassemble
        physical tensors, and push them into the inner
        :class:`PolynomialModel`.

        :param params: parameter tensors matching :attr:`param_names`
            (e.g. ``[K, R, Q, S]`` or ``[K, R, Q, S, B]``)
        :type params: list[torch.Tensor]
        """
        for name, tensor in zip(self.param_names, params):
            setattr(self, name, tensor)
        self.model.update_params(self.assemble_gas_tensors())

    def retract_general_tensors_to_gas_tensors(
        self,
        tensors: list[torch.Tensor],
        margin: float = 1e-3,
    ) -> None:
        r"""
        Retract general polynomial operator tensors ``[A, H]`` onto the GAS
        parameter manifold and **set** the model's GAS parameters.

        ``A`` is first shifted, if necessary, into the open left-half plane,
        :math:`A \leftarrow A - \alpha I` with

        .. math::

            \alpha = \begin{cases}
                0, & \max_i \mathrm{Re}\,\lambda_i(A) < 0,\\[2pt]
                \max_i \mathrm{Re}\,\lambda_i(A) + \texttt{margin}, &
                    \text{otherwise,}
            \end{cases}

        so that the (stabilized) ``A`` is Hurwitz.  The Lyapunov equation

        .. math::

            A^\top P + P A = -I

        is then solved for the SPD :math:`P`, and the GAS parameters are built
        from the skew/symmetric split of :math:`A P^{-1} = N - M`:

        .. math::

            N = \mathrm{skew}(A P^{-1}), \quad
            M = -\mathrm{sym}(A P^{-1}) = \tfrac12 P^{-2} \succ 0, \\
            K = \tfrac12 N, \quad
            R = \mathrm{chol}(M)^{-1}, \quad
            Q = \mathrm{chol}(P^{-1})^\top, \quad
            S_{:,:,k} = \tfrac14 \bigl(H_{:,:,k} P^{-1} - (H_{:,:,k} P^{-1})^\top\bigr),

        which give :math:`K - K^\top = N`, :math:`R^{-1}R^{-\top} = M` and
        :math:`Q^{-1}Q^{-\top} = P`.  The reconstruction of ``A`` is then exact,

        .. math::

            \bigl((K - K^\top) - R^{-1}R^{-\top}\bigr)\,Q^{-1}Q^{-\top}
                = (N - M)\,P = A P^{-1} P = A.

        If the quadratic operator ``H`` is structured (skew-symmetric under the
        :math:`P` metric, i.e., :math:`H_{:,:,k} P^{-1}` is skew-symmetric), then
        its reconstruction is also exact:

        .. math::

            S_{:,:,k} Q^{-1}Q^{-\top} - S_{:,:,k}^\top Q^{-1}Q^{-\top}
                = 2 S_{:,:,k} P = H_{:,:,k}.

        Using the Lyapunov solution :math:`P` (rather than :math:`P = I`)
        guarantees :math:`M` is SPD for *any* Hurwitz ``A``, so spectral
        stability alone suffices -- :math:`\mathrm{sym}(A)` need not be
        negative definite.  Any forcing operator ``B`` is left unchanged.

        :param tensors: ``[A, H]`` with ``A`` of shape ``(r, r)`` and ``H``
            of shape ``(r, r, r)``
        :type tensors: list[torch.Tensor]
        :param margin: stability margin by which the spectrum is pushed into
            the left-half plane when ``A`` is not already strictly stable
        :type margin: float
        :raises RuntimeError: if the assembled ``[K, R, Q, S]`` fail to
            reconstruct the (stabilized) ``A``, or if a structured ``H``
            fails to be reconstructed
        """
        from scipy.linalg import solve_continuous_lyapunov

        A = tensors[0].to(device=self.device, dtype=self.dtype)
        H = tensors[1].to(device=self.device, dtype=self.dtype)
        r = A.shape[0]
        eye = torch.eye(r, device=self.device, dtype=self.dtype)

        # Shift the spectrum into the open left-half plane if A is not already
        # strictly stable (leave it unchanged otherwise).
        abscissa = float(torch.linalg.eigvals(A).real.max())
        shift = abscissa + margin if abscissa >= 0.0 else 0.0
        A = A - shift * eye

        # Solve A^T P + P A = -I for the SPD Lyapunov solution P.
        P_np = solve_continuous_lyapunov(
            A.T.detach().cpu().numpy(), (-eye).detach().cpu().numpy()
        )
        P = torch.as_tensor(P_np, device=self.device, dtype=self.dtype)
        P = 0.5 * (P + P.T)  # symmetrize against round-off
        Pinv = torch.linalg.inv(P)

        # Split A P^{-1} = N - M with N skew and M = -sym(A P^{-1}) = P^{-2}/2 SPD.
        APinv = A @ Pinv
        N = 0.5 * (APinv - APinv.T)
        M = -0.5 * (APinv + APinv.T)

        # K = N / 2 (so K - K^T = N);  R^{-1}R^{-T} = M;  Q^{-1}Q^{-T} = P.
        K = 0.5 * N
        R = torch.linalg.inv(torch.linalg.cholesky(M))
        Q = torch.linalg.cholesky(Pinv).T

        # S_{:,:,k} = skew(H_{:,:,k} P^{-1}): transpose the leading (i, j) axes.
        H_Pinv = torch.einsum("ijk,jl->ilk", H, Pinv)
        S = 0.25 * (H_Pinv - H_Pinv.permute(1, 0, 2))

        # Verify the reconstruction A = ((K - K^T) - R^{-1}R^{-T}) Q^{-1}Q^{-T}.
        Qinv = torch.linalg.inv(Q)
        Rinv = torch.linalg.inv(R)
        A_recon = ((K - K.T) - Rinv @ Rinv.T) @ (Qinv @ Qinv.T)
        err = torch.linalg.norm(A_recon - A) / torch.linalg.norm(A)
        if err > 1e-6:
            raise RuntimeError(
                f"GAS retraction failed to reconstruct A (rel. error {err:.2e})."
            )

        # Verify the reconstruction of H if the input was already structured.
        if 2 in self.poly_comp:
            Qtil = Qinv @ Qinv.T
            H_recon = torch.einsum("ilk,lj->ijk", S, Qtil) - torch.einsum(
                "lik,lj->ijk", S, Qtil
            )
            # Check if the input H was already structured (skew-symmetric under the P metric)
            is_structured = torch.linalg.norm(H_Pinv + H_Pinv.permute(1, 0, 2)) / (
                torch.linalg.norm(H_Pinv) + 1e-12
            ) < 1e-6
            if is_structured:
                err_H = torch.linalg.norm(H_recon - H) / (torch.linalg.norm(H) + 1e-12)
                if err_H > 1e-6:
                    raise RuntimeError(
                        f"GAS retraction failed to reconstruct H (rel. error {err_H:.2e})."
                    )

        # Set the GAS parameters (preserving B and any other current params).
        retracted = {"K": K, "R": R, "Q": Q, "S": S}
        params = [
            retracted.get(name, getattr(self, name)) for name in self.param_names
        ]
        self.update_params(params)

    def evaluate_rhs(self, t: float, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Delegate to the inner :class:`PolynomialModel`."""
        return self.model.evaluate_rhs(t, z, **kwargs)

    def evaluate_adjoint_rhs(
        self, t: float, z: torch.Tensor, Z: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Delegate to the inner :class:`PolynomialModel`."""
        return self.model.evaluate_adjoint_rhs(t, z, Z, **kwargs)

    def vjp_evaluate_rhs(
        self, z: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> list[torch.Tensor]:
        r"""
        VJP of the RHS with respect to the GAS parameters.

        Calls the inner :class:`PolynomialModel` VJP to get gradients
        w.r.t. ``(A, H, [B])``, then propagates through the GAS assembly
        to obtain gradients w.r.t. ``(K, R, Q, S, [B])``.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :type z: torch.Tensor
        :param v: upstream adjoint seed, same shape as ``z``
        :type v: torch.Tensor
        :returns: list of gradients matching :attr:`param_names`
        :rtype: list[torch.Tensor]
        """
        inner_grads = self.model.vjp_evaluate_rhs(z, v, **kwargs)

        # Unpack inner gradients (indexed by position in poly_comp)
        grad_A = inner_grads[self.poly_comp.index(1)] if 1 in self.poly_comp else None
        grad_H = inner_grads[self.poly_comp.index(2)] if 2 in self.poly_comp else None
        grad_B = inner_grads[-1] if self.forcing_exists else None

        Qinv = torch.linalg.inv(self.Q)
        Qtil = Qinv @ Qinv.T

        grads = []

        # grad_K, grad_R (from linear term A = ((K - K^T) - R^{-1} R^{-T}) @ Qtil)
        Rinv = torch.linalg.inv(self.R) if 1 in self.poly_comp else None
        if 1 in self.poly_comp:
            grad_A_Qtil = grad_A @ Qtil
            sym = grad_A_Qtil + grad_A_Qtil.T
            grad_K = grad_A_Qtil - grad_A_Qtil.T
            # M = R^{-1} R^{-T}; chain through M = P P^T and P = R^{-1}.
            grad_R = Rinv.T @ sym @ Rinv @ Rinv.T
            grads.extend([grad_K, grad_R])

        # grad_Q, grad_S from H_{ijk} = (S_{ilk} - S_{lik}) Qtil_{lj}
        if 2 in self.poly_comp:
            # grad_Qtil from linear term: A_pre^T @ grad_A
            if grad_A is not None:
                A_pre = (self.K - self.K.T) - Rinv @ Rinv.T
                grad_Qtil = A_pre.T @ grad_A
            else:
                grad_Qtil = torch.zeros_like(Qtil)
            # grad_Qtil_{lj} from quadratic: Σ_{ik} grad_H_{ijk} (S_{ilk} - S_{lik})
            grad_Qtil += (
                torch.einsum("jik,jlk->il", self.S, grad_H)
                - torch.einsum("ijk,jlk->il", self.S, grad_H)
            )
            # grad_Q from Qtil = Q^{-1} Q^{-T}
            grad_Q = -(Qinv.T @ grad_Qtil @ Qtil + Qinv.T @ grad_Qtil.T @ Qtil)

            # grad_S_{abc} = Σ_j grad_H_{ajc} Qtil_{bj} - Σ_j grad_H_{bjc} Qtil_{aj}
            grad_S = (
                torch.einsum("ijk,jl->ilk", grad_H, Qtil)
                - torch.einsum("jl,ilk->jik", Qtil, grad_H)
            )
            grads.extend([grad_Q, grad_S])

        if self.forcing_exists:
            grads.append(grad_B)

        return grads
