import torch
import torch.nn as nn


class OpInfParams(nn.Module):

    def __init__(self, OpInf_class, initial_guess=None, requires_grad=True):
        super().__init__()

        self.OpInf_class = OpInf_class
        self._r = self.OpInf_class.r
        self._poly_comp = self.OpInf_class.poly_comp
        self._tensor_shapes = [(self._r,) * (d + 1) for d in self._poly_comp]
        self._tensor_names = [f"A{i}" for i in self._poly_comp]

        if initial_guess is not None:
            for i, name in enumerate(self._tensor_names):
                self.register_parameter(
                    name,
                    nn.Parameter(
                        getattr(initial_guess, name),
                        requires_grad=requires_grad,
                    ),
                )
        else:
            dev = self.OpInf_class.Phi.device
            dtype = self.OpInf_class.Phi.dtype
            for i, shape in enumerate(self._tensor_shapes):
                self.register_parameter(
                    self._tensor_names[i],
                    nn.Parameter(
                        torch.randn(*shape, device=dev, dtype=dtype),
                        requires_grad=requires_grad,
                    ),
                )

    @property
    def tensor_names(self):
        return list(self._tensor_names)
    
    def get_tensors(self):
        return [getattr(self, name) for name in self._tensor_names]
    


class GasOpInfParams(nn.Module):

    def __init__(self, OpInf_class, initial_guess=None, requires_grad=True):
        super().__init__()

        if OpInf_class.gas_flag is False:
            raise ValueError (
                f"You cannot call this module if OpInf_class.gas_flag is False."
            )
        self.OpInf_class = OpInf_class
        self._r = self.OpInf_class.r
        self._poly_comp = self.OpInf_class.poly_comp
        self._tensor_shapes = [(self._r, self._r)] * 3 + [(self._r, self._r, self._r)]
        self._tensor_names = ["K", "R", "Q", "S"]

        if initial_guess is not None:
            for i, name in enumerate(self._tensor_names):
                self.register_parameter(
                    name,
                    nn.Parameter(
                        getattr(initial_guess, name),
                        requires_grad=requires_grad,
                    ),
                )
        else:
            dev = self.OpInf_class.Phi.device
            dtype = self.OpInf_class.Phi.dtype
            for i, shape in enumerate(self._tensor_shapes):
                self.register_parameter(
                    self._tensor_names[i],
                    nn.Parameter(
                        torch.randn(*shape, device=dev, dtype=dtype),
                        requires_grad=requires_grad,
                    ),
                )

    @property
    def tensor_names(self):
        return list(self._tensor_names)
    
    def get_tensors(self):
        return [getattr(self, name) for name in self._tensor_names]


class OpInfForwardModule(nn.Module):

    def __init__(
            self,
            OpInf_class,
            OpInf_params: nn.Module
    ):
        super().__init__()
        self.OpInf_class = OpInf_class
        self.OpInf_params = OpInf_params

    def forward(self):
        return self.OpInf_class.cost(self.OpInf_params.get_tensors())

    def gradient(self):
        return self.OpInf_class.gradient(self.OpInf_params.get_tensors())