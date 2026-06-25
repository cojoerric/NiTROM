import torch
from itertools import combinations
from string import ascii_lowercase
from .model import Model


class PolynomialModel(Model):
    r"""
    Class for a polynomial ROM of the form

    .. math::

        f(z, u) = v + Az + H:z z^\top + \ldots + B u,

    where :math:`z` is the state and :math:`u` some external forcing.

    :param r: reduced state dimension
    :type r: int
    :param poly_comp: list of polynomial degrees, e.g. ``[1, 2]`` for linear + quadratic
    :type poly_comp: list[int]
    :param device: device for tensor allocation
    :type device: torch.device or str
    :param dtype: data type for tensors
    :type dtype: torch.dtype
    :param instability_threshold: norm threshold above which the state is considered blown up
    :type instability_threshold: float
    :param tensors: optional list of operator tensors :math:`[A_1, A_2, \ldots]`.
        If ``None``, tensors are initialized to zero.  When
        ``forcing_config`` is set, the last tensor is ``B``.
    :type tensors: list[torch.Tensor] or None
    :param forcing_config: optional dict with keys ``"forcing_exists"``
        (bool) and ``"m"`` (int, forcing input dimension).  When provided
        and ``forcing_exists`` is ``True``, the last entry of ``tensors``
        (or a zero-initialized ``(r, m)`` matrix) is treated as ``B``.
    :type forcing_config: dict or None
    """

    def __init__(
        self,
        r: int,
        poly_comp: list[int],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        instability_threshold: float = 1e6,
        tensors: list[torch.Tensor] = None,
        forcing_config: dict | None = None,
    ):
        forcing_exists = forcing_config is not None and forcing_config.get(
            "forcing_exists", False
        )
        # An optional fixed input operator may be supplied via
        # ``forcing_config["B"]``; B remains a parameter but is flagged
        # non-learnable so that its gradient is zeroed.
        B_fixed = forcing_config.get("B") if forcing_config else None

        param_names = [f"A_{k}" for k in poly_comp]
        if forcing_exists:
            param_names.append("B")
        super().__init__(r, param_names, device, dtype)

        self.poly_comp = poly_comp
        self.thresh = instability_threshold
        self.forcing_exists = forcing_exists

        # Initialize tensors (zero, or the fixed B) if not provided
        if tensors is None:
            tensors = [
                torch.zeros((r,) * (k + 1), device=self.device, dtype=self.dtype)
                for k in poly_comp
            ]
            if forcing_exists:
                if B_fixed is not None:
                    tensors.append(B_fixed.to(device=self.device, dtype=self.dtype))
                else:
                    m = forcing_config["m"]
                    tensors.append(
                        torch.zeros((r, m), device=self.device, dtype=self.dtype)
                    )
        self.update_params(tensors)
        # A supplied fixed B always overrides any value from tensors.
        if forcing_exists and B_fixed is not None:
            self.B = B_fixed.to(device=self.device, dtype=self.dtype)
        self._generate_einsum_subscripts()

    def get_params(self) -> list[torch.Tensor]:
        """Return the current parameter tensors as a list."""
        return [getattr(self, name) for name in self.param_names]

    def _generate_einsum_subscripts(self) -> None:
        """
        Generates the indices for the einsum evaluation of the
        right-hand side and the adjoint
        """
        ss = []
        for k in self.poly_comp:
            ssk = ascii_lowercase[: k + 1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)
        self.einsum_ss = tuple(ss)

    def update_params(self, tensors: list[torch.Tensor]) -> None:
        r"""
        Update the operator tensors of the polynomial model.

        :param tensors: new list of operator tensors :math:`[A_1, A_2, \ldots]`,
            in the same order as :attr:`param_names`
        :type tensors: list[torch.Tensor]
        """
        for name, tensor in zip(self.param_names, tensors):
            setattr(self, name, tensor)

    def evaluate_rhs(self, t: float, z: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Evaluate the ROM right-hand side:

        .. math::

            \dot{z} = \sum_k A_k \underbrace{(z, \ldots, z)}_{k} + B f(t)

        :param t: time instance
        :type t: float
        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :type z: torch.Tensor
        :keyword external_forcing: list of callables :math:`[f_0, f_1, \ldots]` where
            :math:`f_i(t)` returns the forcing for the *i*-th trajectory
        :type external_forcing: list[callable] or None
        :rtype: torch.Tensor
        """

        f_fun_lst = kwargs.get("external_forcing", None)
        tensors = self.get_params()

        # z is a vector
        if z.ndim == 1:
            # Guard against blow-up
            if torch.linalg.vector_norm(z) >= self.thresh:
                return torch.zeros_like(z)

            # Compute the dynamics
            dzdt = torch.zeros_like(z)
            for i, k in enumerate(self.poly_comp):
                equation = ",".join(self.einsum_ss[i])
                operands = [tensors[i]] + [z for _ in range(k)]
                dzdt += torch.einsum(equation, *operands)

            # Add the forcing
            if f_fun_lst is not None:
                f = torch.atleast_1d(f_fun_lst[0](t))
                dzdt += self.B @ f if self.forcing_exists else f

        # z is a tensor (we use batching to evaluate all vectors at once)
        else:
            # Guard against blow-up. If all entries > thresh, then return zeros,
            # otherwise compute the rhs of the vectors that are < thresh
            norms = torch.linalg.vector_norm(z, dim=-1)
            mask = norms < self.thresh
            if not mask.any():
                return torch.zeros_like(z)

            # Compute the dynamics
            dzdt = torch.zeros_like(z)
            for i, k in enumerate(self.poly_comp):
                parts = self.einsum_ss[i]
                eq_parts = [parts[0]] + [f"...{p}" for p in parts[1:]]
                equation = ",".join(eq_parts)
                operands = [tensors[i]] + [z[mask] for _ in range(k)]
                dzdt[mask] += torch.einsum(equation, *operands)

            # Add the external forcing
            if f_fun_lst is not None:
                for i in range(len(f_fun_lst)):
                    if mask[i] and f_fun_lst[i] is not None:
                        f = torch.atleast_1d(f_fun_lst[i](t))
                        dzdt[i] += self.B @ f if self.forcing_exists else f

        return dzdt

    def evaluate_adjoint_rhs(
        self, t: float, z: torch.Tensor, Z: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        r"""
        Evaluate the adjoint right-hand side:

        .. math::

            \dot{z} = J(Z)^\top z

        where :math:`J(Z) = \nabla_z f(Z)` is the Jacobian of the RHS evaluated at
        the base flow :math:`Z`.

        :param t: time instance
        :type t: float
        :param z: adjoint state vector of shape ``(n,)`` or ``(m, n)``
        :type z: torch.Tensor
        :param Z: base flow at which to evaluate the Jacobian, same shape as ``z``
        :type Z: torch.Tensor
        :rtype: torch.Tensor
        """
        n = z.shape[-1]
        tensors = self.get_params()

        # z is a vector
        if z.ndim == 1:
            # Guard against blow-up
            if torch.linalg.vector_norm(z) >= self.thresh:
                return torch.zeros_like(z)

            # Compute the Jacobian and adjoint dynamics
            J = torch.zeros((n, n), device=z.device, dtype=z.dtype)
            for i, k in enumerate(self.poly_comp):
                if k == 0:
                    continue
                combs = list(combinations(self.einsum_ss[i][1:], r=k - 1))
                operands = [tensors[i]] + [Z for _ in range(k - 1)]
                for comb in combs:
                    equation = ",".join([self.einsum_ss[i][0]] + list(comb))
                    J += torch.einsum(equation, *operands)
            dzdt = J.T @ z

        # z is a tensor (we use batching to evaluate all vectors at once)
        else:
            # Guard against blow-up
            norms = torch.linalg.vector_norm(z, dim=-1)
            mask = norms < self.thresh
            if not mask.any():
                return torch.zeros_like(z)

            # Compute the Jacobian and adjoint dynamics
            dzdt = torch.zeros_like(z)
            Jb = torch.zeros((mask.sum(), n, n), device=z.device, dtype=z.dtype)
            for i, k in enumerate(self.poly_comp):
                if k == 0:
                    continue
                combs = list(combinations(self.einsum_ss[i][1:], r=k - 1))
                for comb in combs:
                    eq_parts = [self.einsum_ss[i][0]] + [f"...{p}" for p in comb]
                    equation = ",".join(eq_parts)
                    operands = [tensors[i]] + [Z[mask] for _ in range(k - 1)]
                    Jb += torch.einsum(equation, *operands)
            dzdt[mask] = torch.einsum("bnm,bn->bm", Jb, z[mask])

        return dzdt

    def vjp_evaluate_rhs(
        self, z: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> list[torch.Tensor]:
        r"""
        VJP of :meth:`evaluate_rhs` with respect to the operator tensors
        and, if forcing is present, with respect to :math:`B`.

        For each degree-:math:`k` tensor :math:`A_k`, the gradient is

        .. math::

            \frac{\partial J}{\partial A_k}
                = v \otimes \underbrace{z \otimes \cdots \otimes z}_{k}.

        For the input matrix :math:`B` (where the forward pass contributes
        :math:`B\,u(t)`), the gradient is

        .. math::

            \frac{\partial J}{\partial B} = v\, u(t)^\top.

        For the batched case, contributions are summed over the batch.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :type z: torch.Tensor
        :param v: upstream adjoint seed, same shape as ``z``
        :type v: torch.Tensor
        :keyword external_forcing: list of callables returning the forcing
        :keyword t: time at which to evaluate the forcing (required if
            ``external_forcing`` is provided)
        :returns: list of gradients ``[grad_A_0, ..., grad_A_K, grad_B]``
            where ``grad_B`` is only included when forcing is present
        :rtype: list[torch.Tensor]
        """
        grads = []

        f_fun_lst = kwargs.get("external_forcing", None)
        t = kwargs.get("t", 0.0)

        if z.ndim == 1:
            for i, k in enumerate(self.poly_comp):
                ss = self.einsum_ss[i]
                out_subscript = ss[0]
                in_subscripts = [ss[0][0]] + list(ss[0][1:])
                equation = ",".join(in_subscripts) + "->" + out_subscript
                operands = [v] + [z for _ in range(k)]
                grads.append(torch.einsum(equation, *operands))

            # grad_B = v @ u(t)^T
            if f_fun_lst is not None:
                u = torch.atleast_1d(f_fun_lst[0](t))
                grads.append(torch.outer(v, u))
        else:
            for i, k in enumerate(self.poly_comp):
                ss = self.einsum_ss[i]
                out_subscript = ss[0]
                in_subscripts = [f"...{s}" for s in ss[0]]
                equation = ",".join(in_subscripts) + "->" + out_subscript
                operands = [v] + [z for _ in range(k)]
                grads.append(torch.einsum(equation, *operands))

            # grad_B = sum_j v_j @ u_j(t)^T
            if f_fun_lst is not None:
                grad_B = torch.zeros_like(self.B)
                for j in range(z.shape[0]):
                    u = torch.atleast_1d(f_fun_lst[j](t))
                    grad_B += torch.outer(v[j], u)
                grads.append(grad_B)

        return grads
