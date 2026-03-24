import abc
import torch


class Projection(metaclass=abc.ABCMeta):
    """Abstract base class for projections between full and reduced spaces."""

    @property
    @abc.abstractmethod
    def param_names(self) -> list[str]:
        """
        Names of the projection parameters.

        :rtype: list[str]
        """
        ...

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update the projection parameters.
        Subclasses define the specific arguments required.
        """
        ...

    @abc.abstractmethod
    def encode(self, q: torch.Tensor) -> torch.Tensor:
        """
        Project from full space to reduced space.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :type q: torch.Tensor
        :rtype: torch.Tensor
        """
        ...

    @abc.abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct from reduced space to full space.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :type z: torch.Tensor
        :rtype: torch.Tensor
        """
        ...

    @abc.abstractmethod
    def vjp_encode(self, q: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> tuple:
        """
        VJP of the encoder with respect to the projection parameters.
        Subclasses define the specific parameters and return order.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :type q: torch.Tensor
        :param v: upstream adjoint seed of shape matching the encoder output
        :type v: torch.Tensor
        :rtype: tuple[torch.Tensor, ...]
        """
        ...

    @abc.abstractmethod
    def vjp_decode(self, z: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> tuple:
        """
        VJP of the decoder with respect to the projection parameters.
        Subclasses define the specific parameters and return order.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :type z: torch.Tensor
        :param v: upstream adjoint seed of shape matching the decoder output
        :type v: torch.Tensor
        :rtype: tuple[torch.Tensor, ...]
        """
        ...
