import torch
from ..latent_space_models.polynomial_model import PolynomialModel
from ..projections.linear_projection import LinearProjection


class PolyDynamicsLinearProjection():
    r"""
    A polynomial model where the dynamics are linear in the original state space,
    but the projection to the latent space is nonlinear (polynomial).
    """

    def __init__(self, model: PolynomialModel, proj: LinearProjection):
        self.model = model
        self.proj = proj

    def update(self, params: list[list[torch.Tensor], list[torch.Tensor]]) -> None:
        self.model.update(params[0])
        self.proj.update(params[1])

