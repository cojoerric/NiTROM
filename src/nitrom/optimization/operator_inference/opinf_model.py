import torch
import torch.nn as nn

from ..rom_utils import construct_operators, propagate_gradients


class OpInfModel(nn.Module):
    r"""
    Operator-inference model: holds trainable parameters and computes
    the least-squares cost and analytic gradient.

    Minimizes

    .. math::

        J = \operatorname{tr}(R\,W\,R^\top) + \lambda \lVert H \rVert^2

    where :math:`R = \dot{Z} - A Z - H(Z, Z)`,
    :math:`Z = \Phi^\top X`, and :math:`\dot{Z} = \Phi^\top \dot{X}`.

    :param opt_obj: training data object with ``X``, ``dX``, ``weights``
    :type opt_obj: TrainingData
    :param poly_comp: polynomial degrees, e.g. ``[1, 2]``
    :type poly_comp: list[int]
    :param Phi: trial basis of shape ``(N, r)``
    :type Phi: torch.Tensor
    :param reg: Tikhonov regularization for ``H``
    :type reg: float
    :param initial_guess: optional list of tensors to initialize the
        parameters.  Must match the expected number: ``len(poly_comp)``
        for standard mode (``[A1, A2, ...]``), or 4 for GAS mode
        (``[K, R, Q, S]``).  If ``None``, parameters are initialized
        randomly.
    :type initial_guess: list[torch.Tensor] or None
    :param gas_flag: if ``True``, use the GAS parameterization
    :type gas_flag: bool
    """

    def __init__(
        self,
        opt_obj,
        poly_comp: list[int],
        Phi: torch.Tensor,
        reg: float = 0.0,
        initial_guess: list[torch.Tensor]=None,
        gas_flag: bool = False,
    ) -> None:
        super().__init__()

        self.poly_comp = poly_comp
        self.Phi = Phi
        self.reg = reg
        self.gas_flag = gas_flag

        ntraj, _, nt = opt_obj.X.shape
        self.ntraj = ntraj
        self.nt = nt
        self.r = Phi.shape[-1]

        dev = Phi.device
        dtype = Phi.dtype

        # Precompute projected data
        self.Z = torch.einsum("ij,kil->kjl", Phi, opt_obj.X)
        self.dZ = torch.einsum("ij,kil->kjl", Phi, opt_obj.dX)
        W = (1 / opt_obj.weights).repeat_interleave(nt)
        self.W = torch.diag(W)

        # Register trainable parameters
        if not gas_flag:
            self._param_names = [f"A{k}" for k in poly_comp]
            shapes = [(self.r,) * (k + 1) for k in poly_comp]
        else:
            self._param_names = ["K", "R", "Q", "S"]
            shapes = [
                (self.r, self.r),
                (self.r, self.r),
                (self.r, self.r),
                (self.r, self.r, self.r),
            ]

        if initial_guess is not None:
            if len(initial_guess) != len(self._param_names):
                raise ValueError(
                    f"initial_guess has {len(initial_guess)} tensors, "
                    f"expected {len(self._param_names)}"
                )
            for name, tensor in zip(self._param_names, initial_guess):
                self.register_parameter(
                    name,
                    nn.Parameter(tensor.to(device=dev, dtype=dtype)),
                )
        else:
            for name, shape in zip(self._param_names, shapes):
                self.register_parameter(
                    name,
                    nn.Parameter(torch.randn(*shape, device=dev, dtype=dtype)),
                )

    @property
    def tensor_names(self) -> list[str]:
        """Names of the trainable parameters."""
        return list(self._param_names)

    def get_tensors(self) -> list[torch.Tensor]:
        """Return the current parameter tensors as a list."""
        return [getattr(self, name) for name in self._param_names]

    def forward(self) -> torch.Tensor:
        r"""
        Evaluate the weighted least-squares cost.

        :returns: scalar loss
        :rtype: torch.Tensor
        """
        params = self.get_tensors()
        if self.gas_flag:
            params, _ = construct_operators(params, self.poly_comp)
        A, H = params

        AZ = torch.matmul(A, self.Z)
        HZZ = torch.einsum("ijk,bjt,bkt->bit", H, self.Z, self.Z)
        R = self.dZ - (AZ + HZZ)
        R_flat = R.permute(1, 0, 2).reshape(self.r, self.ntraj * self.nt)
        return ((R_flat @ self.W) * R_flat).sum() + self.reg * torch.norm(H) ** 2

    def gradient(self) -> list[torch.Tensor]:
        r"""
        Compute the analytic gradient of the cost with respect to the
        trainable parameters.

        :returns: list of gradient tensors, one per parameter
        :rtype: list[torch.Tensor]
        """
        params = self.get_tensors()
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
            grads = propagate_gradients(
                grads, other_tensors, raw_params, self.poly_comp
            )

        return grads
