from typing import Any

from .model import Model
from .polynomial_model import PolynomialModel

import numpy as np


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
    :param device: device for array allocation (ignored by the NumPy backend)
    :type device: str
    :param dtype: data type for arrays; defaults to the backend's ``float64``
    :type dtype: backend dtype or None
    :param instability_threshold: norm threshold for blow-up guard
    :type instability_threshold: float
    :param gas_params: optional list of initial GAS parameter tensors
        (subset depending on ``poly_comp``).
        If ``None``, parameters are initialized randomly.
    :type gas_params: list or None
    :param forcing_config: optional dict with keys ``"forcing_exists"``
        (bool) and ``"m"`` (int).  See :class:`PolynomialModel`.
    :type forcing_config: dict or None
    """

    def __init__(
        self,
        r: int,
        poly_comp: list[int],
        device: str = "cpu",
        dtype: Any = None,
        instability_threshold: float = 1e6,
        gas_params: list | None = None,
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
        bkend = self.backend
        self.poly_comp = poly_comp
        self.forcing_exists = forcing_exists

        # Set GAS parameters as attributes
        if gas_params is not None:
            for name, tensor in zip(param_names, gas_params, strict=True):
                setattr(
                    self, name,
                    bkend.asarray(tensor, dtype=self.dtype, device=self.device),
                )
        else:
            # Initialize the GAS parameters randomly (K, R, Q, S; B handled below)
            gas_param_names = [name for name in param_names if name != "B"]
            for name, shape in zip(gas_param_names, gas_shapes, strict=True):
                setattr(
                    self, name,
                    bkend.randn(shape, dtype=self.dtype, device=self.device),
                )
            # Initialize B (fixed value if supplied, else zeros)
            if forcing_exists:
                m = forcing_config["m"]
                self.B = (
                    bkend.asarray(B_fixed, dtype=self.dtype, device=self.device)
                    if B_fixed is not None
                    else bkend.zeros((r, m), dtype=self.dtype, device=self.device)
                )

        # A supplied fixed B always overrides any value from gas_params.
        if forcing_exists and B_fixed is not None:
            self.B = bkend.asarray(B_fixed, dtype=self.dtype, device=self.device)

        # Assemble physical tensors and create the inner PolynomialModel
        tensors = self.assemble_gas_tensors()
        self.model = PolynomialModel(
            r,
            poly_comp,
            device=device,
            dtype=self.dtype,
            instability_threshold=instability_threshold,
            tensors=tensors,
            forcing_config=forcing_config,
        )

    def get_params(self) -> list[Any]:
        """Return the current GAS parameter tensors as a list."""
        return [getattr(self, name) for name in self.param_names]

    def assemble_gas_tensors(self) -> list[Any]:
        r"""
        Build physical operator tensors from the current GAS parameters.

        :returns: list of tensors ``[A, H, ..., B]`` matching the inner
            :class:`PolynomialModel` param order.  ``B`` is appended
            only when forcing is present.
        :rtype: list
        """
        bkend = self.backend
        tensors = [None] * len(self.poly_comp)

        Qinv = bkend.inv(self.Q)
        Qtil = Qinv @ Qinv.T

        if 1 in self.poly_comp:
            idx = self.poly_comp.index(1)
            Rinv = bkend.inv(self.R)
            tensors[idx] = ((self.K - self.K.T) - Rinv @ Rinv.T) @ Qtil

        if 2 in self.poly_comp:
            idx = self.poly_comp.index(2)
            tensors[idx] = bkend.einsum(
                "ilk,lj->ijk", self.S, Qtil
            ) - bkend.einsum("lik,lj->ijk", self.S, Qtil)

        if self.forcing_exists:
            tensors.append(self.B)

        return tensors

    def update_params(self, params: list) -> None:
        r"""
        Update the GAS parameters (and B if present), reassemble
        physical tensors, and push them into the inner
        :class:`PolynomialModel`.

        :param params: parameter tensors matching :attr:`param_names`
            (e.g. ``[K, R, Q, S]`` or ``[K, R, Q, S, B]``)
        :type params: list
        """
        for name, tensor in zip(self.param_names, params, strict=True):
            setattr(self, name, tensor)
        self.model.update_params(self.assemble_gas_tensors())

    def retract_general_tensors_to_gas_tensors(
        self,
        tensors: list,
        margin: float = 1e-3,
        lyapunov_P: bool = True,
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
        :type tensors: list
        :param margin: stability margin by which the spectrum is pushed into
            the left-half plane when ``A`` is not already strictly stable
        :type margin: float
        :raises RuntimeError: if the assembled ``[K, R, Q, S]`` fail to
            reconstruct the (stabilized) ``A``, or if a structured ``H``
            fails to be reconstructed
        """
        from scipy.linalg import solve_continuous_lyapunov

        bkend = self.backend
        A = bkend.asarray(tensors[0], dtype=self.dtype, device=self.device)
        H = bkend.asarray(tensors[1], dtype=self.dtype, device=self.device)
        r = A.shape[0]
        eye = bkend.eye(r, dtype=self.dtype, device=self.device)

        # Shift the spectrum to guarantee stability (and positive definiteness of M)
        if lyapunov_P:
            # Shift the spectrum into the open left-half plane if A is not already
            # strictly stable (leave it unchanged otherwise).
            abscissa = float(bkend.eigvals(A).real.max())
            shift = abscissa + margin if abscissa >= 0.0 else 0.0
            A = A - shift * eye
        else:
            # Shift the spectrum of sym(A) into the negative definite plane to guarantee M is SPD when P = I
            sym_A = 0.5 * (A + A.T)
            abscissa_sym = float(bkend.eigvals(sym_A).real.max())
            shift = abscissa_sym + margin if abscissa_sym >= 0.0 else 0.0
            A = A - shift * eye

        if lyapunov_P:
            # Solve A^T P + P A = -I for the SPD Lyapunov solution P.
            P_np = solve_continuous_lyapunov(
                bkend.to_numpy(A.T), bkend.to_numpy(-eye)
            )
            P = bkend.asarray(P_np, dtype=self.dtype, device=self.device)
        else:
            P = eye

        print(np.linalg.cond(P))
        P = 0.5 * (P + P.T)  # symmetrize against round-off
        Pinv = bkend.inv(P)

        # Split A P^{-1} = N - M with N skew and M = -sym(A P^{-1}) = P^{-2}/2 SPD.
        APinv = A @ Pinv
        N = 0.5 * (APinv - APinv.T)
        M = -0.5 * (APinv + APinv.T)

        # K = N / 2 (so K - K^T = N);  R^{-1}R^{-T} = M;  Q^{-1}Q^{-T} = P.
        K = 0.5 * N
        R = bkend.inv(bkend.cholesky(M))
        Q = bkend.cholesky(Pinv).T

        # S_{:,:,k} = skew(H_{:,:,k} P^{-1}): transpose the leading (i, j) axes.
        H_Pinv = bkend.einsum("ijk,jl->ilk", H, Pinv)
        S = 0.25 * (H_Pinv - bkend.permute(H_Pinv, (1, 0, 2)))

        # Verify the reconstruction A = ((K - K^T) - R^{-1}R^{-T}) Q^{-1}Q^{-T}.
        Qinv = bkend.inv(Q)
        Rinv = bkend.inv(R)
        A_recon = ((K - K.T) - Rinv @ Rinv.T) @ (Qinv @ Qinv.T)
        err = float(bkend.vector_norm(A_recon - A) / bkend.vector_norm(A))
        if err > 1e-6:
            raise RuntimeError(
                f"GAS retraction failed to reconstruct A (rel. error {err:.2e})."
            )

        # Verify the reconstruction of H if the input was already structured.
        if 2 in self.poly_comp:
            Qtil = Qinv @ Qinv.T
            H_recon = bkend.einsum("ilk,lj->ijk", S, Qtil) - bkend.einsum(
                "lik,lj->ijk", S, Qtil
            )
            # Check if the input H was already structured (skew-symmetric under
            # the P metric).
            is_structured = float(
                bkend.vector_norm(H_Pinv + bkend.permute(H_Pinv, (1, 0, 2)))
                / (bkend.vector_norm(H_Pinv) + 1e-12)
            ) < 1e-6
            if is_structured:
                err_H = float(
                    bkend.vector_norm(H_recon - H)
                    / (bkend.vector_norm(H) + 1e-12)
                )
                if err_H > 1e-6:
                    raise RuntimeError(
                        f"GAS retraction failed to reconstruct H "
                        f"(rel. error {err_H:.2e})."
                    )

        # Set the GAS parameters (preserving B and any other current params).
        retracted = {"K": K, "R": R, "Q": Q, "S": S}
        params = [
            retracted.get(name, getattr(self, name)) for name in self.param_names
        ]
        self.update_params(params)

    def evaluate_rhs(self, t: float, z: Any, **kwargs) -> Any:
        """Delegate to the inner :class:`PolynomialModel`."""
        return self.model.evaluate_rhs(t, z, **kwargs)

    def evaluate_adjoint_rhs(self, t: float, z: Any, Z: Any, **kwargs) -> Any:
        """Delegate to the inner :class:`PolynomialModel`."""
        return self.model.evaluate_adjoint_rhs(t, z, Z, **kwargs)

    def vjp_evaluate_rhs(self, z: Any, v: Any, **kwargs) -> list[Any]:
        r"""
        VJP of the RHS with respect to the GAS parameters.

        Calls the inner :class:`PolynomialModel` VJP to get gradients
        w.r.t. ``(A, H, [B])``, then propagates through the GAS assembly
        to obtain gradients w.r.t. ``(K, R, Q, S, [B])``.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :param v: upstream adjoint seed, same shape as ``z``
        :returns: list of gradients matching :attr:`param_names`
        :rtype: list
        """
        bkend = self.backend
        inner_grads = self.model.vjp_evaluate_rhs(z, v, **kwargs)

        # Unpack inner gradients (indexed by position in poly_comp)
        grad_A = inner_grads[self.poly_comp.index(1)] if 1 in self.poly_comp else None
        grad_H = inner_grads[self.poly_comp.index(2)] if 2 in self.poly_comp else None
        grad_B = inner_grads[-1] if self.forcing_exists else None

        Qinv = bkend.inv(self.Q)
        Qtil = Qinv @ Qinv.T

        grads = []

        # grad_K, grad_R (from linear term A = ((K - K^T) - R^{-1} R^{-T}) @ Qtil)
        Rinv = bkend.inv(self.R) if 1 in self.poly_comp else None
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
                grad_Qtil = bkend.zeros_like(Qtil)
            # grad_Qtil_{lj} from quadratic: Σ_{ik} grad_H_{ijk} (S_{ilk} - S_{lik})
            grad_Qtil += bkend.einsum(
                "jik,jlk->il", self.S, grad_H
            ) - bkend.einsum("ijk,jlk->il", self.S, grad_H)
            # grad_Q from Qtil = Q^{-1} Q^{-T}
            grad_Q = -(Qinv.T @ grad_Qtil @ Qtil + Qinv.T @ grad_Qtil.T @ Qtil)

            # grad_S_{abc} = Σ_j grad_H_{ajc} Qtil_{bj} - Σ_j grad_H_{bjc} Qtil_{aj}
            grad_S = bkend.einsum("ijk,jl->ilk", grad_H, Qtil) - bkend.einsum(
                "jl,ilk->jik", Qtil, grad_H
            )
            grads.extend([grad_Q, grad_S])

        if self.forcing_exists:
            grads.append(grad_B)

        return grads
