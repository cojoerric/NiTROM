from collections.abc import Callable
from typing import Any, Literal

from nitrom.backend import get_backend
from nitrom.utils import interp_quadratic


def _jacobian_f(f, t_eval, y, args, kwargs, bkend):
    """Jacobian of ``f(t_eval, .)`` at ``y`` -- ``(n, n)`` or batched ``(B, n, n)``.

    Uses ``torch.func.jacrev`` (exact) on the torch backend and a forward
    finite difference on the numpy backend.
    """
    n = y.shape[-1]
    dev = bkend.device_of(y)

    if bkend.is_torch:
        import torch
        from torch.func import jacrev

        y_detached = y.detach().requires_grad_(True)

        def single_f(u):
            return f(t_eval, u, *args, **kwargs)

        if y.ndim == 1:
            return jacrev(single_f)(y_detached).detach()

        B_size = y.shape[0]
        J = bkend.zeros((B_size, n, n), dtype=y.dtype, device=dev)
        for b in range(B_size):
            def single_f_batched(u, b=b):
                y_list = [
                    y_detached[i].detach() if i != b else u
                    for i in range(B_size)
                ]
                return f(t_eval, torch.stack(y_list, dim=0), *args, **kwargs)[b]
            J[b] = jacrev(single_f_batched)(y_detached[b]).detach()
        return J

    # numpy: forward finite difference (columns of J), batched together.
    eps = 1e-7
    f0 = f(t_eval, y, *args, **kwargs)
    if y.ndim == 1:
        J = bkend.zeros((n, n), dtype=y.dtype, device=dev)
        for j in range(n):
            yp = bkend.copy(y)
            yp[j] += eps
            J[:, j] = (f(t_eval, yp, *args, **kwargs) - f0) / eps
        return J

    B_size = y.shape[0]
    J = bkend.zeros((B_size, n, n), dtype=y.dtype, device=dev)
    for j in range(n):
        yp = bkend.copy(y)
        yp[:, j] += eps
        # f is batched and each output row depends only on its own input row.
        J[:, :, j] = (f(t_eval, yp, *args, **kwargs) - f0) / eps
    return J


def _newton_solve(
    f: Callable[..., Any],
    t_eval: float,
    y0: Any,
    rhs_const: Any,
    dt: float,
    alpha: float,
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    *args,
    **kwargs,
) -> Any:
    r"""
    Solve the implicit system for y:

        y - rhs_const - dt * alpha * f(t_eval, y, *args, **kwargs) = 0

    using Newton-Raphson.
    """
    bkend = get_backend()
    y = bkend.copy(y0)
    n = y.shape[-1]

    eye = bkend.eye(n, dtype=y.dtype, device=bkend.device_of(y))
    if y.ndim == 2:
        eye = eye[None]  # (1, n, n) for batched broadcasting

    res_norm = None
    for _ in range(newton_max_iter):
        f_val = f(t_eval, y, *args, **kwargs)
        F_val = y - rhs_const - dt * alpha * f_val

        if y.ndim == 1:
            res_norm = bkend.vector_norm(F_val)
        else:
            res_norm = bkend.vector_norm(F_val, axis=-1).max()
        res_val = float(res_norm.detach()) if hasattr(res_norm, "detach") else float(res_norm)
        if res_val < newton_tol:
            break

        J_f = _jacobian_f(f, t_eval, y, args, kwargs, bkend)
        J_F = eye - (dt * alpha) * J_f
        delta_y = bkend.solve(J_F, -F_val)
        y = y + delta_y
    else:
        res_val = float(res_norm.detach()) if hasattr(res_norm, "detach") else float(res_norm)
        raise RuntimeError(
            f"Newton solver failed to converge within {newton_max_iter} "
            f"iterations. Final residual norm: {res_val:.2e}"
        )

    return y


def evolve(
    f: Callable[..., Any],
    t: float,
    x: Any,
    dt: float,
    method: Literal["rk4", "rk2", "backward_euler"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    *args,
    **kwargs,
) -> Any:
    r"""
    Advance the state by one step using an explicit or implicit Runge-Kutta method.

    :param f: right-hand side :math:`f(t, x, \ldots)`
    :param t: current time
    :param x: current state of shape ``(n,)`` or ``(B, n)``
    :param dt: time-step size
    :param method: ``"rk4"``, ``"rk2"``, or ``"backward_euler"``
    :param newton_tol: tolerance for the Newton solver (implicit methods only)
    :param newton_max_iter: max iterations for the Newton solver (implicit only)
    :param args: extra positional arguments forwarded to *f*
    :param kwargs: extra keyword arguments forwarded to *f*
    :returns: state at time :math:`t + \Delta t`
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
        # Predictor step: forward Euler for a good initial guess.
        y0 = x + dt * f(t, x, *args, **kwargs)
        x_next = _newton_solve(
            f, t + dt, y0, x, dt, 1.0,
            newton_tol, newton_max_iter, *args, **kwargs,
        )
    else:
        raise ValueError(f"Unknown integration method: {method}")
    return x_next


def solve_ivp(
    f: Callable[..., Any],
    x0: Any,
    t0: float,
    tf: float,
    dt: float,
    t_eval: Any,
    method: Literal["rk4", "rk2", "backward_euler"] = "rk4",
    newton_tol: float = 1e-8,
    newton_max_iter: int = 20,
    *args,
    **kwargs,
) -> Any:
    r"""
    Integrate an ODE IVP with the specified Runge-Kutta method and return the
    solution interpolated at the requested evaluation times.

    The integrator steps on a uniform grid with spacing close to *dt*
    (adjusted so that :math:`t_f - t_0` is an exact multiple), stores the
    solution at a sub-sampled rate, and interpolates onto *t_eval*.

    :param f: right-hand side :math:`f(t, x, \ldots)`
    :param x0: initial condition of shape ``(n,)`` or ``(B, n)``
    :param t0: initial time
    :param tf: final time
    :param dt: desired time-step size (will be adjusted slightly)
    :param t_eval: times at which to return the solution, shape ``(n_eval,)``
    :param method: ``"rk4"``, ``"rk2"``, or ``"backward_euler"``
    :param newton_tol: tolerance for the Newton solver (implicit methods only)
    :param newton_max_iter: max iterations for the Newton solver (implicit only)
    :param args: extra positional arguments forwarded to *f*
    :param kwargs: extra keyword arguments forwarded to *f*
    :returns: solution of shape ``(n, n_eval)`` or ``(B, n, n_eval)``
    """
    bkend = get_backend()
    dev = bkend.device_of(x0)
    dtype = x0.dtype
    batched = x0.ndim == 2
    t0, tf, dt = float(t0), float(tf), float(dt)

    # Time points for the simulation.
    nt_sim = int(round((tf - t0) / dt))
    dt = (tf - t0) / nt_sim
    tsim = dt * bkend.arange(nt_sim + 1, dtype=dtype, device=dev) + t0
    assert abs(float(tf) - float(tsim[-1])) < 1e-10

    # Time points for saving: store at a uniform sub-sample, then interpolate.
    dteval_min = float((t_eval[1:] - t_eval[:-1]).min())
    save_every = max(int(round(dteval_min / dt)), 1)
    tsave = tsim[::save_every]
    n_save = len(tsave)

    x = bkend.copy(x0)
    if batched:
        B, n = x0.shape
        X = bkend.zeros((B, n, n_save), dtype=dtype, device=dev)
        X[:, :, 0] = x
    else:
        n = x0.shape[0]
        X = bkend.zeros((n, n_save), dtype=dtype, device=dev)
        X[:, 0] = x

    for i in range(1, len(tsim)):
        t = tsim[i - 1]
        x = evolve(
            f, t, x, dt, method, newton_tol, newton_max_iter,
            *args, **kwargs,
        )
        if i % save_every == 0:
            if batched:
                X[:, :, i // save_every] = x
            else:
                X[:, i // save_every] = x

    return interp_quadratic(t_eval, tsave, X)


def solve_adjoint_ivp_discrete(
    f: Callable[..., Any],
    vjp_z: Callable[..., Any],
    vjp_theta: Callable[..., Any],
    Zint: Any,
    sub_t: Any,
    h: float,
    lam_init: Any,
    method: Literal["rk4", "rk2"] = "rk4",
    *args,
    **kwargs,
) -> tuple[Any, list[Any]]:
    r"""
    Propagate the adjoint state backward in time through the discrete RK solver
    stages and accumulate the parameter VJPs.

    :param f: right-hand side evaluate_rhs: ``(t, z, *args, **kwargs) -> dz``
    :param vjp_z: VJP w.r.t state (evaluate_adjoint_rhs): ``(t, w, z, *args, **kwargs) -> dz_vjp``
    :param vjp_theta: VJP w.r.t parameters: ``(z, w, t=t, *args, **kwargs) -> list_of_theta_vjps``
    :param Zint: forward trajectory states over the sub-grid: ``(B, n, n_substeps + 1)``
    :param sub_t: time values of the sub-grid: ``(n_substeps + 1,)``
    :param h: step size
    :param lam_init: initial adjoint seed state: ``(B, n)``
    :param method: ``"rk4"`` or ``"rk2"``
    :returns: tuple containing the final adjoint state and the accumulated parameter gradients list
    """
    bkend = get_backend()
    lam = bkend.copy(lam_init) if hasattr(lam_init, "copy") else lam_init * 1.0
    
    n_substeps = len(sub_t) - 1
    param_grads = None
    
    for j in range(n_substeps - 1, -1, -1):
        z_n = Zint[:, :, j]
        t_n = float(sub_t[j])
        
        if method == "rk4":
            k1 = f(t_n, z_n, *args, **kwargs)
            k2 = f(t_n + h / 2.0, z_n + (h / 2.0) * k1, *args, **kwargs)
            k3 = f(t_n + h / 2.0, z_n + (h / 2.0) * k2, *args, **kwargs)
            
            x4 = z_n + h * k3
            x3 = z_n + (h / 2.0) * k2
            x2 = z_n + (h / 2.0) * k1
            x1 = z_n
            
            lam_x = bkend.copy(lam) if hasattr(lam, "copy") else lam * 1.0
            w4 = lam * (h / 6.0)
            w3 = lam * (h / 3.0)
            w2 = lam * (h / 3.0)
            w1 = lam * (h / 6.0)
            
            # Stage 4
            dz4 = vjp_z(t_n + h, w4, x4, *args, **kwargs)
            dtheta4 = vjp_theta(x4, w4, t=t_n + h, *args, **kwargs)
            lam_x = lam_x + dz4
            w3 = w3 + h * dz4
            if param_grads is None:
                param_grads = [bkend.zeros_like(p) for p in dtheta4]
            for idx, g in enumerate(dtheta4):
                param_grads[idx] = param_grads[idx] + g
                
            # Stage 3
            dz3 = vjp_z(t_n + h / 2.0, w3, x3, *args, **kwargs)
            dtheta3 = vjp_theta(x3, w3, t=t_n + h / 2.0, *args, **kwargs)
            lam_x = lam_x + dz3
            w2 = w2 + (h / 2.0) * dz3
            for idx, g in enumerate(dtheta3):
                param_grads[idx] = param_grads[idx] + g
                
            # Stage 2
            dz2 = vjp_z(t_n + h / 2.0, w2, x2, *args, **kwargs)
            dtheta2 = vjp_theta(x2, w2, t=t_n + h / 2.0, *args, **kwargs)
            lam_x = lam_x + dz2
            w1 = w1 + (h / 2.0) * dz2
            for idx, g in enumerate(dtheta2):
                param_grads[idx] = param_grads[idx] + g
                
            # Stage 1
            dz1 = vjp_z(t_n, w1, x1, *args, **kwargs)
            dtheta1 = vjp_theta(x1, w1, t=t_n, *args, **kwargs)
            lam_x = lam_x + dz1
            for idx, g in enumerate(dtheta1):
                param_grads[idx] = param_grads[idx] + g
                
            lam = lam_x
            
        elif method == "rk2":
            k1 = f(t_n, z_n, *args, **kwargs)
            x2 = z_n + h * k1
            x1 = z_n
            
            lam_x = bkend.copy(lam) if hasattr(lam, "copy") else lam * 1.0
            w2 = lam * (h / 2.0)
            w1 = lam * (h / 2.0)
            
            # Stage 2
            dz2 = vjp_z(t_n + h, w2, x2, *args, **kwargs)
            dtheta2 = vjp_theta(x2, w2, t=t_n + h, *args, **kwargs)
            lam_x = lam_x + dz2
            w1 = w1 + h * dz2
            if param_grads is None:
                param_grads = [bkend.zeros_like(p) for p in dtheta2]
            for idx, g in enumerate(dtheta2):
                param_grads[idx] = param_grads[idx] + g
                
            # Stage 1
            dz1 = vjp_z(t_n, w1, x1, *args, **kwargs)
            dtheta1 = vjp_theta(x1, w1, t=t_n, *args, **kwargs)
            lam_x = lam_x + dz1
            for idx, g in enumerate(dtheta1):
                param_grads[idx] = param_grads[idx] + g
                
            lam = lam_x
            
        else:
            raise NotImplementedError(f"Discrete adjoint not implemented for {method}")
            
    return lam, param_grads
