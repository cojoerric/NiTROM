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

    :param poly_comp: list of polynomial degrees, e.g. ``[1, 2]`` for linear + quadratic
    :type poly_comp: list[int]
    :param tensors: list of operator tensors :math:`[A_1, A_2, \ldots]` with shapes
        :math:`(n,)^{k+1}`
    :type tensors: list[torch.Tensor]
    :param instability_threshold: norm threshold above which the state is considered blown up
    :type instability_threshold: float
    """

    def __init__(
        self,
        poly_comp: list[int],
        tensors: list[torch.Tensor],
        instability_threshold: float = 1e6,
        B: torch.Tensor=None,
    ):
        super().__init__()
        self.poly_comp = poly_comp
        self.tensors = tensors
        self.thresh = instability_threshold
        self._param_names = [f"A_{k}" for k in poly_comp]

        self.B = B
        self.forcing_exists = True if B is not None else False

    @property
    def param_names(self) -> list[str]:
        """
        Get the names of the model parameters.

        :rtype: list[str]
        """
        return self._param_names

    
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

    def update(self, tensors: list[torch.Tensor]) -> None:
        r"""
        Update the operator tensors of the polynomial model.

        :param tensors: new list of operator tensors :math:`[A_1, A_2, \ldots]`
        :type tensors: list[torch.Tensor]
        """
        self.tensors = tensors

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

        # z is a vector
        if z.ndim == 1:
            # Guard against blow-up
            if torch.linalg.vector_norm(z) >= self.thresh:
                return torch.zeros_like(z)

            # Compute the dynamics
            dzdt = torch.zeros_like(z)
            for i, k in enumerate(self.poly_comp):
                equation = ",".join(self.einsum_ss[i])
                operands = [self.tensors[i]] + [z for _ in range(k)]
                dzdt += torch.einsum(equation, *operands)

            # Add the forcing
            if f_fun_lst is not None:
                dzdt += self.B @ f_fun_lst[0](t)

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
                operands = [self.tensors[i]] + [z[mask] for _ in range(k)]
                dzdt[mask] += torch.einsum(equation, *operands)

            # Add the external forcing
            if f_fun_lst is not None:
                for i in range(len(f_fun_lst)):
                    if mask[i] and f_fun_lst[i] is not None:
                        dzdt[i] += self.B @ f_fun_lst[i](t)

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
                operands = [self.tensors[i]] + [Z for _ in range(k - 1)]
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
                    operands = [self.tensors[i]] + [Z[mask] for _ in range(k - 1)]
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
                u = f_fun_lst[0](t)
                grads.append(torch.outer(v, u))
        else:
            norms = torch.linalg.vector_norm(z, dim=-1)
            mask = norms < self.thresh
            z_stable = z[mask]
            v_stable = v[mask]

            for i, k in enumerate(self.poly_comp):
                ss = self.einsum_ss[i]
                out_subscript = ss[0]
                in_subscripts = [f"...{s}" for s in ss[0]]
                equation = ",".join(in_subscripts) + "->" + out_subscript
                operands = [v_stable] + [z_stable for _ in range(k)]
                grads.append(torch.einsum(equation, *operands))

            # grad_B = sum_j v_j @ u_j(t)^T (over stable batch items)
            if f_fun_lst is not None:
                idxs = mask.nonzero(as_tuple=False).squeeze(-1)
                grad_B = torch.zeros_like(self.B)
                for j in idxs:
                    u = f_fun_lst[j](t)
                    grad_B += torch.outer(v[j], u)
                grads.append(grad_B)

        return grads
