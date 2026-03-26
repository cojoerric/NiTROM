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

        A = \bigl((K - K^\top) - R\,R^\top\bigr)\,\tilde{Q},
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

        # Track forcing
        forcing_exists = forcing_config is not None and forcing_config.get(
            "forcing_exists", False
        )
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
            # Initialize GAS params randomly
            for name, shape in zip(
                param_names[:-1] if forcing_exists else param_names, gas_shapes
            ):
                setattr(
                    self, name, torch.randn(shape, device=self.device, dtype=self.dtype)
                )
            # Initialize B to zeros
            if forcing_exists:
                m = forcing_config["m"]
                self.B = torch.zeros((r, m), device=self.device, dtype=self.dtype)

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
            tensors[idx] = ((self.K - self.K.T) - self.R @ self.R.T) @ Qtil

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

        # grad_K, grad_R (from linear term A = ((K - K^T) - R R^T) @ Qtil)
        if 1 in self.poly_comp:
            grad_K = grad_A @ Qtil - Qtil @ grad_A.T
            grad_R = -(grad_A @ Qtil + Qtil @ grad_A.T) @ self.R
            grads.extend([grad_K, grad_R])

        # grad_Q, grad_S from H_{ijk} = (S_{ilk} - S_{lik}) Qtil_{lj}
        if 2 in self.poly_comp:
            # grad_Qtil from linear term: A_pre^T @ grad_A
            A_pre = (self.K - self.K.T) - self.R @ self.R.T
            grad_Qtil = A_pre.T @ grad_A if grad_A is not None else torch.zeros_like(Qtil)
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
