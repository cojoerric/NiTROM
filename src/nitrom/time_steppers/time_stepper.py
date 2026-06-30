from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from torch.func import jacrev, vmap

from nitrom.utils import interp_quadratic


def _newton_solve(
    f: Callable[..., torch.Tensor],
    t_eval: float,
    y0: torch.Tensor,
    rhs_const: torch.Tensor,
    dt: float,
    alpha: float,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    *args,
    **kwargs,
) -> torch.Tensor:
    r"""
    Solve the implicit system of equations for y:

        y - rhs_const - dt * alpha * f(t_eval, y, *args, **kwargs) = 0

    using Newton-Raphson.
    """
    y = y0.clone()
    n = y.shape[-1]

    # Pre-allocate identity matrix
    I = torch.eye(n, device=y.device, dtype=y.dtype)
    if y.ndim == 2:
        I = I.unsqueeze(0)  # (1, n, n) for batched broadcasting

    for _ in range(newton_max_iter):
        f_val = f(t_eval, y, *args, **kwargs)
        F_val = y - rhs_const - dt * alpha * f_val

        if y.ndim == 1:
            res_norm = torch.linalg.norm(F_val)
        else:
            res_norm = torch.linalg.norm(F_val, dim=-1).max()
        if res_norm < newton_tol:
            break

        # Compute Jacobian of f with respect to y.
        # We detach y to avoid tracking second-order derivatives in the autograd
        # graph, which is much more memory/time efficient and remains mathematically
        # exact at convergence under the Implicit Function Theorem.
        y_detached = y.detach().requires_grad_(True)

        def single_f(u):
            return f(t_eval, u, *args, **kwargs)

        if y.ndim == 1:
            J_f = jacrev(single_f)(y_detached).detach()
            J_F = I - (dt * alpha) * J_f
        else:
            B_size = y.shape[0]
            J_f = torch.zeros((B_size, n, n), device=y.device, dtype=y.dtype)
            for b in range(B_size):
                yb_detached = y_detached[b]
                # To handle batched inputs to f (e.g. adjoint RHS involving batched base states),
                # we define a function that places the single vector u back into the batch at
                # index b, evaluates the batched function, and returns only the b-th output.
                # We construct y_temp out-of-place using torch.stack to be fully compatible with
                # PyTorch's functional AD transforms (jacrev) which do not support in-place updates.
                def single_f_batched(u):
                    y_list = [y_detached[i].detach() if i != b else u for i in range(B_size)]
                    y_temp = torch.stack(y_list, dim=0)
                    return f(t_eval, y_temp, *args, **kwargs)[b]
                J_f[b] = jacrev(single_f_batched)(yb_detached).detach()
            J_F = I - (dt * alpha) * J_f

        delta_y = torch.linalg.solve(J_F, -F_val)
        y = y + delta_y
    else:
        raise RuntimeError(
            f"Newton solver failed to converge within {newton_max_iter} iterations. "
            f"Final residual norm: {res_norm.item():.2e}"
        )

    return y


def evolve(
    f: Callable[..., torch.Tensor],
    t: float,
    x: torch.Tensor,
    dt: float,
    method: Literal["rk4", "rk2", "backward_euler"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    *args,
    **kwargs,
) -> torch.Tensor:
    r"""
    Advance the state by one time step using an explicit or implicit Runge-Kutta method.

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
    :param method: integration scheme, ``"rk4"``, ``"rk2"``, or ``"backward_euler"``
    :type method: str
    :param newton_tol: tolerance for Newton-Raphson solver (implicit methods only)
    :type newton_tol: float
    :param newton_max_iter: maximum iterations for Newton-Raphson solver (implicit methods only)
    :type newton_max_iter: int
    :param args: extra positional arguments forwarded to *f*
    :param kwargs: extra keyword arguments forwarded to *f* (e.g.
        ``external_forcing``)
    :returns: state at time :math:`t + \Delta t`
    :rtype: torch.Tensor
    """
    if method == "rk4":
        k1 = f(t, x, *args, **kwargs)
        k2 = f(t + dt / 2, x + dt / 2 * k1, *args, **kwargs)
        k3 = f(t + dt / 2, x + dt / 2 * k2, *args, **kwargs)
        k4 = f(t + dt, x + dt * k3, *args, **kwargs)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    elif method == "rk2":
        k1 = f(t, x, *args, **kwargs)
        k2 = f(t + dt, x + dt * k1, *args, **kwargs)
        x_next = x + dt / 2 * (k1 + k2)
    elif method == "backward_euler":
        # Predictor step: use Forward Euler to get a good initial guess
        y0 = x + dt * f(t, x, *args, **kwargs)
        x_next = _newton_solve(
            f,
            t + dt,
            y0,
            x,
            dt,
            1.0,
            newton_tol,
            newton_max_iter,
            *args,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown integration method: {method}")
    return x_next


def solve_ivp(
    f: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    t0: float,
    tf: float,
    dt: float,
    t_eval: torch.Tensor,
    method: Literal["rk4", "rk2", "backward_euler"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    *args,
    **kwargs,
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
    :param method: integration scheme, ``"rk4"``, ``"rk2"``, or ``"backward_euler"``
    :type method: str
    :param newton_tol: tolerance for Newton-Raphson solver (implicit methods only)
    :type newton_tol: float
    :param newton_max_iter: maximum iterations for Newton-Raphson solver (implicit methods only)
    :type newton_max_iter: int
    :param args: extra positional arguments forwarded to *f*
    :param kwargs: extra keyword arguments forwarded to *f* (e.g.
        ``external_forcing``)
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
    save_every = max(int(torch.round(dteval_min / dt).item()), 1)
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
        x = evolve(
            f, t, x, dt, method,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
            *args, **kwargs
        )
        if i % save_every == 0:
            if batched:
                X[:, :, i // save_every] = x
            else:
                X[:, i // save_every] = x

    return interp_quadratic(t_eval, tsave, X)


