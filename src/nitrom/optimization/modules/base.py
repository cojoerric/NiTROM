import abc

import torch
import torch.nn as nn


class InferenceModule(nn.Module, abc.ABC):
    r"""
    Abstract base class for inference modules trained with analytic gradients.

    An inference module is an :class:`torch.nn.Module` that exposes a scalar
    cost via :meth:`forward` and the *analytic* gradient of that cost via
    :meth:`gradient`.  Keeping both on the same object lets a single training
    loop (see :func:`nitrom.optimization.train`) drive any concrete module --
    operator inference, polynomial-manifold inference, NiTROM, ... -- without
    relying on autograd.

    **Invariant.** :meth:`gradient` must return one tensor per trainable
    parameter, in the *same order* as :meth:`torch.nn.Module.parameters`.
    The training loop zips the two together to assign ``param.grad``, so the
    ordering contract must hold for every subclass.
    """

    @abc.abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Evaluate the scalar cost.

        :returns: scalar loss
        :rtype: torch.Tensor
        """
        ...

    @abc.abstractmethod
    def gradient(self) -> list[torch.Tensor]:
        """
        Compute the analytic gradient of the cost with respect to the
        trainable parameters, in the same order as :meth:`parameters`.

        :returns: list of gradient tensors, one per trainable parameter
        :rtype: list[torch.Tensor]
        """
        ...
