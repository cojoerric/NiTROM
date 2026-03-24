import torch
from .rom_utils import construct_operators, propagate_gradients


class OpInfCostAndGrad:
    r"""
    Operator-inference cost function and gradient.

    Solves the least-squares problem

    .. math::

        \min_{A, H} \sum_k w_k \lVert \dot{Z}_k - A Z_k - H(Z_k, Z_k) \rVert^2
        + \lambda \lVert H \rVert^2

    where :math:`Z = \Phi^\top X` are the projected snapshots and
    :math:`\dot{Z} = \Phi^\top \dot{X}` the projected derivatives.

    :param opt_obj: training data object containing ``X``, ``dX``, and ``weights``
    :type opt_obj: TrainingData
    :param poly_comp: polynomial composition, e.g. ``[1, 2]``
    :type poly_comp: list[int]
    :param Phi: trial basis of shape ``(N, r)``
    :type Phi: torch.Tensor
    :param reg: Tikhonov regularization coefficient for ``H``
    :type reg: float
    :param gas_flag: if ``True``, reconstruct operators from a
        GAS-parameterization before evaluating
    :type gas_flag: bool
    :param initial_guess: optional initial guess for the optimizer
    :type initial_guess: torch.Tensor or None
    """

    def __init__(
        self,
        opt_obj,
        poly_comp: list[int],
        Phi: torch.Tensor,
        reg: float = 0.0,
        gas_flag: bool = False,
        initial_guess: torch.Tensor | None = None,
    ) -> None:
        self.opt_obj = opt_obj
        self.poly_comp = poly_comp
        self.Phi = Phi
        self.reg = reg
        self.gas_flag = gas_flag
        self.initial_guess = initial_guess

        self.ntraj, self.n, self.nt = self.opt_obj.X.shape
        self.r = self.Phi.shape[-1]

        dev = self.Phi.device
        dtype = self.Phi.dtype
        if self.initial_guess == None:
            if not self.gas_flag:
                A = torch.rand((self.r, self.r), device=dev, dtype=dtype)
                H = torch.rand((self.r, self.r, self.r), device=dev, dtype=dtype)
                initial_guess = [A, H]
            else:
                K = torch.rand((self.r, self.r), device=dev, dtype=dtype)
                R = torch.rand((self.r, self.r), device=dev, dtype=dtype)
                Q = torch.rand((self.r, self.r), device=dev, dtype=dtype)
                S = torch.rand((self.r, self.r, self.r), device=dev, dtype=dtype)
                initial_guess = [K, R, Q, S]
            self.initial_guess = initial_guess

        self.Z = torch.einsum('ij,kil->kjl', self.Phi, opt_obj.X)
        self.dZ = torch.einsum('ij,kil->kjl', self.Phi, opt_obj.dX)
        W = (1 / opt_obj.weights).repeat_interleave(self.nt)
        self.W = torch.diag(W)

    def cost(self, params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""
        Evaluate the weighted least-squares cost.

        .. math::

            J = \operatorname{tr}\!\bigl(R\,W\,R^\top\bigr)
            + \lambda \lVert H \rVert^2

        where :math:`R = \dot{Z} - A Z - H(Z, Z)`.

        :param params: ``(A, H)`` or a GAS-parameterized vector if
            ``gas_flag`` is ``True``
        :type params: tuple[torch.Tensor, ...]
        :returns: scalar cost
        :rtype: torch.Tensor
        """
        if self.gas_flag:
            params, _ = construct_operators(params, self.poly_comp)
        A, H = params

        AZ = torch.matmul(A, self.Z)
        HZZ = torch.einsum("ijk,bjt,bkt->bit", H, self.Z, self.Z)
        R = self.dZ - (AZ + HZZ)
        R_flat = R.permute(1, 0, 2).reshape(self.r, self.ntraj * self.nt)
        loss = ((R_flat @ self.W) * R_flat).sum() + self.reg * torch.norm(H) ** 2

        return loss

    def gradient(self, params: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        r"""
        Compute the gradient of the cost with respect to ``(A, H)``.

        .. math::

            \nabla_A J = -2\, R\,W\, Z^\top, \qquad
            \nabla_H J = -2\, R\,W\,(Z \otimes Z)^\top + 2\lambda H

        If ``gas_flag`` is ``True``, the gradients are propagated through
        the GAS parameterization.

        :param params: ``(A, H)`` or a GAS-parameterized vector
        :type params: tuple[torch.Tensor, ...]
        :returns: gradients in the same format as *params*
        :rtype: tuple[torch.Tensor, ...]
        """
        if self.gas_flag:
            raw_params = params
            params, other_tensors = construct_operators(params, self.poly_comp)
        A, H = params

        AZ = torch.matmul(A, self.Z)
        HZZ = torch.einsum("ijk,bjt,bkt->bit", H, self.Z, self.Z)
        R = self.dZ - (AZ + HZZ)
        R_flat = R.permute(1, 0, 2).reshape(self.r, self.ntraj * self.nt)
        RW = (R_flat @ self.W).reshape(self.r, self.ntraj, self.nt).permute(1, 0, 2)

        grad_A = -2 * torch.einsum("bit,bjt->ij", RW, self.Z)
        grad_H = -2 * torch.einsum("bit,bjt,bkt->ijk", RW, self.Z, self.Z)
        grad_H += 2 * self.reg * H
        grads = (grad_A, grad_H)

        if self.gas_flag:
            grads = propagate_gradients(grads, other_tensors, raw_params, self.poly_comp)

        return grads
