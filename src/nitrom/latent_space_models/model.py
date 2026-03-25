import torch
import torch.nn as nn
import abc


class Model(metaclass=abc.ABCMeta):
    r"""Abstract base class for NiTROM

    :param r: reduced state dimension
    :type r: int
    :param device: device for tensor allocation
    :type device: torch.device or str
    :param dtype: data type for tensors
    :type dtype: torch.dtype
    """

    def __init__(
        self,
        r: int,
        param_names: list[str],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self._r = r
        self._device = torch.device(device)
        self._dtype = dtype
        self._param_names = param_names

    @property
    def state_dimension(self) -> int:
        """Reduced state dimension."""
        return self._r

    @property
    def device(self) -> torch.device:
        """Device on which tensors are allocated."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of tensors."""
        return self._dtype

    @property
    def param_names(self) -> list[str]:
        """Names of the model parameters."""
        return self._param_names

    @abc.abstractmethod
    def get_params(self) -> list[torch.Tensor]:
        """
        Return the current parameter tensors as a list, in the same
        order as :attr:`param_names`.

        :rtype: list[torch.Tensor]
        """
        ...

    @abc.abstractmethod
    def update_params(self, *args, **kwargs) -> None:
        """
        Update the model parameters.
        Subclasses define the specific arguments required.
        """
        ...

    @abc.abstractmethod
    def evaluate_rhs(self, t: float, z: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Evaluate the right-hand side of the ROM.
        t:  time instance
        z:  state vector (n,) or (m, n)
        """
        ...

    @abc.abstractmethod
    def evaluate_adjoint_rhs(
        self, t: float, z: torch.Tensor, Z: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        r"""
        Evaluate the adjoint right-hand side.
        t:  time instance
        z:  adjoint state vector (n,) or (m, n)
        Z:  base flow at which to evaluate the Jacobian, same shape as z
        """
        ...

    @abc.abstractmethod
    def vjp_evaluate_rhs(
        self, z: torch.Tensor, v: torch.Tensor, *args, **kwargs
    ) -> list:
        r"""
        VJP of the RHS with respect to the model parameters.
        Subclasses define the specific return contents.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :type z: torch.Tensor
        :param v: upstream adjoint seed, same shape as ``z``
        :type v: torch.Tensor
        :rtype: list[torch.Tensor]
        """
        ...
