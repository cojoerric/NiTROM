import torch

from .projection import Projection


class LinearProjection(Projection):
    r"""
    Oblique linear projection defined by trial basis :math:`\Phi` and
    test basis :math:`\Psi`:

    .. math::

        \text{encode}(q) = \Psi^\top q, \qquad
        \text{decode}(z) = \Phi (\Psi^\top \Phi)^{-1} z

    :param bases: list of two tensors ``[Phi, Psi]``, each of shape ``(N, r)``
    :type bases: list[torch.Tensor]
    """

    def __init__(self, bases: list[torch.Tensor]):
        Phi = bases[0]
        n, r = Phi.shape
        super().__init__(
            n, r, param_names=["Phi", "Psi"],
            device=Phi.device, dtype=Phi.dtype,
        )
        self.Phi = bases[0]
        self.Psi = bases[1]
        self.S = torch.linalg.inv(self.Psi.T @ self.Phi)

    def get_params(self) -> list[torch.Tensor]:
        """Return ``[Phi, Psi]``."""
        return [self.Phi, self.Psi]

    def update(self, params: list[torch.Tensor]) -> None:
        r"""
        Update the trial and test bases and recompute :math:`S = (\Psi^\top \Phi)^{-1}`.

        :param params: list of two tensors ``[Phi, Psi]``, each of shape ``(N, r)``
        :type params: list[torch.Tensor]
        """
        self.Phi = params[0]
        self.Psi = params[1]
        self.S = torch.linalg.inv(self.Psi.T @ self.Phi)

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        r"""
        Project from full space to reduced space: :math:`z = \Psi^\top q`.

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :type q: torch.Tensor
        :rtype: torch.Tensor
        """
        if q.ndim == 1:
            return self.Psi.T @ q
        return (self.Psi.T @ q.T).T

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Reconstruct from reduced space to full space:
        :math:`q = \Phi (\Psi^\top \Phi)^{-1} z`.

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :type z: torch.Tensor
        :rtype: torch.Tensor
        """
        if z.ndim == 1:
            return self.Phi @ (self.S @ z)
        return (self.Phi @ (self.S @ z.T)).T

    def vjp_encode(
        self, q: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        VJP of the encoder :math:`z = \Psi^\top q` with respect to :math:`\Psi`.

        .. math::

            \frac{\partial J}{\partial \Psi} = q\, v^\top

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :type q: torch.Tensor
        :param v: upstream adjoint seed :math:`v = \partial J / \partial z`
            of shape ``(r,)`` or ``(m, r)``
        :type v: torch.Tensor
        :returns: ``(grad_Phi, grad_Psi)``
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if q.ndim == 1:
            grad_Psi = torch.outer(q, v)
        else:
            # q is (m, N), v is (m, r) -> q.T @ v sums over batch -> (N, r)
            grad_Psi = q.T @ v
        return (torch.zeros_like(grad_Psi), grad_Psi)

    def vjp_decode(
        self, z: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        VJP of the decoder :math:`\hat{q} = \Phi\, S\, z` with respect to
        :math:`\Phi` and :math:`\Psi`, where :math:`S = (\Psi^\top \Phi)^{-1}`.

        .. math::

            \frac{\partial J}{\partial \Phi}
                = \bigl(v - \Psi\, S^\top \Phi^\top v\bigr)(S\,z)^\top,
            \qquad
            \frac{\partial J}{\partial \Psi}
                = -(\Phi\, S\, z)(S^\top \Phi^\top v)^\top

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :type z: torch.Tensor
        :param v: upstream adjoint seed :math:`v = \partial J / \partial \hat{q}`
            of shape ``(N,)`` or ``(m, N)``
        :type v: torch.Tensor
        :returns: ``(grad_Phi, grad_Psi)``
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if v.ndim == 1:
            # Unbatched: v is (N,), z is (r,)
            w = torch.outer(v, self.S @ z)                      # (N, r)
        else:
            # Batched: v is (m, N), z is (m, r) -> v.T @ (z @ S.T) -> (N, r)
            w = v.T @ (z @ self.S.T)                             # (N, r)
        # grad_Phi = w - Psi @ S^T @ Phi^T @ w
        PhiTw = self.Phi.T @ w                                   # (r, r)
        grad_Phi = w - self.Psi @ (self.S.T @ PhiTw)             # (N, r)
        # grad_Psi = -Phi @ w^T @ Phi @ S
        wTPhi = w.T @ self.Phi                                   # (r, r)
        grad_Psi = -self.Phi @ (wTPhi @ self.S)                  # (N, r)
        return (grad_Phi, grad_Psi)