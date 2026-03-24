import torch
import torch.nn as nn
import abc


class Model(metaclass=abc.ABCMeta):
    r"""Abstract base class for NiTROM"""

    @property
    @abc.abstractmethod
    def param_names(self) -> list[str]:
        """
        Names of the model parameters.

        :rtype: list[str]
        """
        ...

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        r"""
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
