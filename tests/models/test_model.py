"""Tests for the abstract Model class."""

import pytest
import torch
from nitrom.latent_space_models.model import Model


# Helper mixin fragments to avoid repetition
_PARAM_NAMES = """
    @property
    def param_names(self):
        return []
"""


class TestModelAbstract:
    """Verify that Model cannot be instantiated and enforces the interface."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Model()

    def test_must_implement_param_names(self):
        class Incomplete(Model):
            def update(self, *args, **kwargs):
                pass

            def evaluate_rhs(self, t, z, **kwargs):
                return torch.zeros_like(z)

            def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
                return torch.zeros_like(z)

            def vjp_evaluate_rhs(self, z, v):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_update(self):
        class Incomplete(Model):
            @property
            def param_names(self):
                return []

            def evaluate_rhs(self, t, z, **kwargs):
                return torch.zeros_like(z)

            def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
                return torch.zeros_like(z)

            def vjp_evaluate_rhs(self, z, v):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_evaluate_rhs(self):
        class Incomplete(Model):
            @property
            def param_names(self):
                return []

            def update(self, *args, **kwargs):
                pass

            def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
                return torch.zeros_like(z)

            def vjp_evaluate_rhs(self, z, v):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_evaluate_adjoint_rhs(self):
        class Incomplete(Model):
            @property
            def param_names(self):
                return []

            def update(self, *args, **kwargs):
                pass

            def evaluate_rhs(self, t, z, **kwargs):
                return torch.zeros_like(z)

            def vjp_evaluate_rhs(self, z, v):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_vjp_evaluate_rhs(self):
        class Incomplete(Model):
            @property
            def param_names(self):
                return []

            def update(self, *args, **kwargs):
                pass

            def evaluate_rhs(self, t, z, **kwargs):
                return torch.zeros_like(z)

            def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
                return torch.zeros_like(z)

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_instantiates(self):
        class Complete(Model):
            @property
            def param_names(self):
                return []

            def update(self, *args, **kwargs):
                pass

            def evaluate_rhs(self, t, z, **kwargs):
                return torch.zeros_like(z)

            def evaluate_adjoint_rhs(self, t, z, Z, **kwargs):
                return torch.zeros_like(z)

            def vjp_evaluate_rhs(self, z, v):
                return []

        model = Complete()
        z = torch.randn(4)
        assert model.param_names == []
        assert model.evaluate_rhs(0.0, z).shape == z.shape
        assert model.evaluate_adjoint_rhs(0.0, z, z).shape == z.shape
