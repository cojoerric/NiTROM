import abc
import torch


class Projection(metaclass=abc.ABCMeta):
    """Abstract base class for projections between full and reduced spaces."""

    def __init__(
        self,
        n: int,
        r: int,
        param_names: list[str],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self._n = n
        self._r = r
        self._device = torch.device(device)
        self._dtype = dtype
        self._param_names = param_names

    @property
    def ambient_space_dimension(self) -> int:
        """Ambient space dimension."""
        return self._n

    @property
    def latent_space_dimension(self) -> int:
        """Ambient space dimension."""
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

    @abc.abstractmethod
    def vjp_decode_state(
        self, z: torch.Tensor, v: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        VJP of the decoder with respect to the latent **state** :math:`z`
        (not the parameters):

        .. math::

            \left(\frac{\partial\,\text{decode}(z)}{\partial z}\right)^\top v.

        This is the decoder Jacobian-transpose applied to a full-space
        cotangent, used to map an ambient-space adjoint seed back into the
        latent space (e.g. the measurement source of an adjoint solve).

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :type z: torch.Tensor
        :param v: full-space cotangent of shape ``(N,)`` or ``(m, N)``
        :type v: torch.Tensor
        :returns: latent-space vector of shape ``(r,)`` or ``(m, r)``
        :rtype: torch.Tensor
        """
        ...
