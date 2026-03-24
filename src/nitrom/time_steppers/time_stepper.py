from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from nitrom.utils import interp_quadratic

def evolve(
    f: Callable[..., torch.Tensor],
    t: float,
    x: torch.Tensor,
    dt: float,
    method: Literal["rk4", "rk2"] = "rk4",
    *args,
) -> torch.Tensor:
    r"""
    Advance the state by one time step using an explicit Runge-Kutta method.

    .. math::

        x_{n+1} = x_n + \Delta t \sum_i b_i k_i

    :param f: right-hand side :math:`f(t, x, \ldots)`
    :type f: Callable[..., torch.Tensor]
    :param t: current time
    :type t: float
    :param x: current state of shape ``(n,)`` or ``(B, n)``
    :type x: torch.Tensor
    :param dt: time-step size
    :type dt: float
    :param method: integration scheme, ``"rk4"`` (classic 4th-order) or
        ``"rk2"`` (Heun's method)
    :type method: str
    :param args: extra positional arguments forwarded to *f*
    :returns: state at time :math:`t + \Delta t`
    :rtype: torch.Tensor
    """
    if method == "rk4":
        k1 = f(t, x, *args)
        k2 = f(t + dt / 2, x + dt / 2 * k1, *args)
        k3 = f(t + dt / 2, x + dt / 2 * k2, *args)
        k4 = f(t + dt, x + dt * k3, *args)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    elif method == "rk2":
        k1 = f(t, x, *args)
        k2 = f(t + dt, x + dt * k1, *args)
        x_next = x + dt / 2 * (k1 + k2)
    return x_next

def solve_ivp(
    f: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    t0: float,
    tf: float,
    dt: float,
    t_eval: torch.Tensor,
    method: Literal["rk4", "rk2"] = "rk4",
    *args,
) -> torch.Tensor:
    r"""
    Integrate an ODE initial-value problem using the specified Runge-Kutta method and return the
    solution interpolated at the requested evaluation times.

    The integrator steps on a uniform grid with spacing close to *dt*
    (adjusted so that :math:`t_f - t_0` is an exact multiple), stores
    the solution at a sub-sampled rate, and interpolates onto
    *t_eval*.

    :param f: right-hand side :math:`f(t, x, \ldots)`
    :type f: Callable[..., torch.Tensor]
    :param x0: initial condition of shape ``(n,)`` or ``(B, n)``
    :type x0: torch.Tensor
    :param t0: initial time
    :type t0: float
    :param tf: final time
    :type tf: float
    :param dt: desired time-step size (will be adjusted slightly)
    :type dt: float
    :param t_eval: times at which to return the solution, shape ``(n_eval,)``
    :type t_eval: torch.Tensor
    :param args: extra positional arguments forwarded to *f*
    :returns: solution tensor of shape ``(n, n_eval)`` or ``(B, n, n_eval)``
    :rtype: torch.Tensor
    """
    dev = x0.device
    dtype = x0.dtype
    batched = x0.ndim == 2

    # Compute the time points for the simulation
    nt_sim = int(np.round((tf - t0) / dt))
    dt = (tf - t0) / nt_sim
    tsim = dt * torch.arange(nt_sim + 1, device=dev, dtype=dtype) + t0
    assert torch.abs(tf - tsim[-1]).item() < 1e-10

    # Compute the time points for saving the solution
    # We do so by storing the solution at uniformly spaced time points,
    # and then we linearly interpolate the solution at the requested
    # time points.
    dteval_min = torch.min(t_eval[1:] - t_eval[:-1])
    save_every = int(torch.round(dteval_min / dt).item())
    tsave = tsim[::save_every]
    n_save = len(tsave)

    x = x0.clone()
    if batched:
        B, n = x0.shape
        X = torch.zeros((B, n, n_save), dtype=dtype, device=dev)
        X[:, :, 0] = x
    else:
        n = x0.shape[0]
        X = torch.zeros((n, n_save), dtype=dtype, device=dev)
        X[:, 0] = x

    for i in range(1, len(tsim)):
        t = tsim[i - 1]
        x = evolve(f, t, x, dt, method, *args)
        if i % save_every == 0:
            if batched:
                X[:, :, i // save_every] = x
            else:
                X[:, i // save_every] = x
    
    return interp_quadratic(t_eval, tsave, X)

