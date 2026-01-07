import torch
import torch.nn as nn
from . import opinf_functions


class OpinfParams(nn.Module):
    def __init__(self, pool, r, poly_comp, init=None, requires_grad=True):
        super().__init__()
        self._r = r
        self._poly_comp = poly_comp
        self.tensor_shapes_from_poly_comp()
        self._tensor_names = []

        for i, shape in enumerate(self._tensor_shapes):
            name = f"A{i+2}"
            self.register_parameter(
                name,
                nn.Parameter(torch.randn(*shape, device=pool.device, dtype=pool.dtype), requires_grad=requires_grad),
            )
            self._tensor_names.append(name)

        if init is not None:
            self._apply_init(init)

    def _apply_init(self, init):
        with torch.no_grad():
            for name in self._tensor_names:
                tensor0 = init.get(name, None)
                if tensor0 is not None:
                    getattr(self, name).copy_(tensor0)

    def tensors(self):
        return [getattr(self, name) for name in self._tensor_names]
    
    def tensor_names(self):
        return list(self._tensor_names)

    def param_tuple(self):
        return tuple(self.tensors())
    
    def tensor_shapes_from_poly_comp(self):
        self._tensor_shapes = []
        for deg in self._poly_comp:
            self._tensor_shapes.append((self._r,)*(deg+1))


class OpinfParams_GloballyStable(nn.Module):
    def __init__(self, pool, r, poly_comp, init=None, requires_grad=True):
        super().__init__()
        self._r = r
        self._poly_comp = poly_comp

        if any(poly_comp_i > 2 for poly_comp_i in poly_comp):
            raise ValueError("Global stability implementation only supports polynomial components up to degree 2.")
        self._tensor_names = ["Qhat"]
        self.Qhat = nn.Parameter(torch.randn(r, r, device=pool.device, dtype=pool.dtype), requires_grad=requires_grad)
        
        if 1 in poly_comp:
            _linear_tensor_names = ["Jhat", "Rhat"]
            self._tensor_names.extend(_linear_tensor_names)
            shape = (r, r)
            for name in _linear_tensor_names:
                self.register_parameter(
                    name,
                    nn.Parameter(torch.randn(*shape, device=pool.device, dtype=pool.dtype), requires_grad=requires_grad),
                )
        if 2 in poly_comp:
            self._tensor_names.append("Hhat")
            shape = (r, r, r)
            self.register_parameter(
                "Hhat",
                nn.Parameter(torch.randn(*shape, device=pool.device, dtype=pool.dtype), requires_grad=requires_grad),
            )
        
        if init is not None:
            self._apply_init(init)
    
    def _apply_init(self, init):
        with torch.no_grad():
            for name in self._tensor_names:
                tensor0 = init.get(name, None)
                if tensor0 is not None:
                    getattr(self, name).copy_(tensor0)

    def tensors(self):
        return [getattr(self, name) for name in self._tensor_names]
    
    def tensor_names(self):
        return list(self._tensor_names)
    
    def param_tuple(self):
        return tuple(self.tensors())
    

class OpinfModel(nn.Module):
    def __init__(
            self,
            phi,
            params: nn.Module,
            opt_obj,
            *,
            regularization_H: float = 0.0,
            return_numpy:bool = False,
    ):
        super().__init__()
        self.name = 'opinf'
        self.params = params
        if any(poly_comp_i > 2 for poly_comp_i in params._poly_comp):
            raise ValueError("Operator inference only supports polynomial components up to degree 2.")
        if "Qhat" in params.tensor_names():
            self.glob_stable = True
        else:
            self.glob_stable = False

        self.cost_fn, self.grad_fn = opinf_functions.create_objective_and_gradient(
            opt_obj,
            phi,
            glob_stable=self.glob_stable,
            poly_comp=params._poly_comp,
            regularization_H=regularization_H,
            return_numpy=return_numpy,
        )

    def forward(self):
        return self.cost_fun(*self.param_tuple())
    
    def param_tuple(self):
        return self.params.param_tuple()
    
    def euclidean_grads(self):
        return self.grad_fn(*self.param_tuple())