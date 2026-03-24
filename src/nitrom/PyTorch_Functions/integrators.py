import torch
import time


def rk4_step(fun, t, x, dt, args=()):
    """
    Performs a single integration step using the fourth-order Runge-Kutta (RK4) method.

    Supports x of shape (n,) or (B, n). fun(t, x, *args) must return the same shape as x.
    """

    k1 = fun(t, x, *args)
    k2 = fun(t + 0.5 * dt, x + 0.5 * dt * k1, *args)
    k3 = fun(t + 0.5 * dt, x + 0.5 * dt * k2, *args)
    k4 = fun(t + dt, x + dt * k3, *args)
    x_new = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_new


def my_rk4(fun, t_vec, x0, args=(), *, n_substeps: int = 10):
    """
    Integrates ODEs using fixed-step RK4 (fast, GPU-friendly).

    Supports:
      - x0: (n,) -> returns (n, T)
      - x0: (B, n) -> returns (B, n, T)

    fun(t, x, *args) must support batched x.

    n_substeps:
      Number of RK4 substeps per interval [t_vec[i-1], t_vec[i]].
      (Keeps Python loop counts constant -> avoids torch->CPU sync from .item()).
    """

    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")

    x = x0.clone()
    Tlen = t_vec.shape[0]

    if x0.ndim == 1:
        xs = torch.zeros((x0.shape[0], Tlen), dtype=x0.dtype, device=x0.device)
        xs[:, 0] = x0
    elif x0.ndim == 2:
        B, n = x0.shape
        xs = torch.zeros((B, n, Tlen), dtype=x0.dtype, device=x0.device)
        xs[:, :, 0] = x0
    else:
        raise ValueError("x0 must be 1D (n,) or 2D (B, n).")

    t = t_vec[0]
    for i in range(1, Tlen):
        T = t_vec[i]
        dt = (T - t) / n_substeps  # tensor scalar on-device

        # If t_vec is non-increasing, just record the current state.
        if torch.any(dt <= 0):
            if x0.ndim == 1:
                xs[:, i] = x
            else:
                xs[:, :, i] = x
            t = T
            continue

        for _ in range(n_substeps):
            x = rk4_step(fun, t, x, dt, args)
            t = t + dt

        if x0.ndim == 1:
            xs[:, i] = x
        else:
            xs[:, :, i] = x

    return xs


def my_rk4_adaptive(
    fun,
    t_vec,
    x0,
    args=(),
    *,
    atol=1e-6,
    rtol=1e-3,
    safety_factor=0.8,
    fac_min=0.1,
    fac_max=5.0,
    n_substeps=1,
):
    """
    Adaptive RK4 with shared step across batch.

    Supports:
      - x0: (n,) -> returns (n, T)
      - x0: (B, n) -> returns (B, n, T)

    fun(t, x, *args) must support batched x.
    """

    x = x0.clone().detach()
    dt = (t_vec[1] - t_vec[0]) / 10  # 0-dim tensor
    Tlen = t_vec.shape[0]

    if x0.ndim == 1:
        xs = torch.zeros((x0.shape[0], Tlen), dtype=x0.dtype, device=x0.device)
        xs[:, 0] = x0
    elif x0.ndim == 2:
        B, n = x0.shape
        xs = torch.zeros((B, n, Tlen), dtype=x0.dtype, device=x0.device)
        xs[:, :, 0] = x0
    else:
        raise ValueError("x0 must be 1D (n,) or 2D (B, n).")

    # Pre-create scalars on the right device/dtype to avoid per-iteration allocations
    one = torch.ones((), device=x0.device, dtype=x0.dtype)
    exponent = 1.0 / 5.0
    err_floor = torch.as_tensor(1e-12, device=x0.device, dtype=x0.dtype)

    t = t_vec[0]
    for i in range(1, Tlen):
        T = t_vec[i]
        while (t < T).item():
            dt_trial = torch.minimum(dt, T - t)

            x_full = rk4_step(fun, t, x, dt_trial, args)
            dt_half = dt_trial * 0.5
            x_half1 = rk4_step(fun, t, x, dt_half, args)
            x_half2 = rk4_step(fun, t + dt_half, x_half1, dt_half, args)

            dx = x_half2 - x_full
            scale = atol + rtol * torch.maximum(torch.abs(x_full), torch.abs(x_half2))
            err = torch.amax(torch.abs(dx) / scale)  # scalar

            if (err <= 1.0).item():
                t = t + dt_trial
                x = x_half2

            # Timestep controller (avoid err==0 branch + avoid new tensor allocations)
            err_safe = torch.maximum(err, err_floor)
            dt_growth = dt_trial * safety_factor * (one / err_safe) ** exponent
            dt = torch.clamp(dt_growth, min=dt * fac_min, max=dt * fac_max)

        if x0.ndim == 1:
            xs[:, i] = x
        else:
            xs[:, :, i] = x

    return xs


def etdrk4_setup(linop, dt):
    """
    Prepares the ETDRK4 matrices for a given linear operator and time step.

    Args:
        linop (tuple): A tuple containing the eigenvalues and eigenvectors of the linear operator.
        dt (float): The time step for the integration.

    Returns:
        E, E2, phi, L_inv3, L_sq: Matrices used in the ETDRK4 method.
    """

    n = linop[0].shape[0]  # Assuming linop is a tuple (V, D, V_inv)

    V, D, V_inv = linop
    L = V @ torch.diag(D) @ V_inv
    E = V @ torch.diag(torch.exp(D * dt)) @ V_inv
    E2 = V @ torch.diag(torch.exp(D * dt / 2)) @ V_inv
    phi = V @ torch.diag(D ** (-1) * (torch.exp(D * dt / 2) - 1)) @ V_inv
    L_inv3 = V @ torch.diag(D ** (-3)) @ V_inv
    L_sq = V @ torch.diag(D**2) @ V_inv

    I = torch.eye(n, device=V.device, dtype=V.dtype)

    coef1 = -4 * I - L * dt + E @ (4 * I - 3 * L * dt + L_sq * dt**2)
    coef2 = 2 * (2 * I + L * dt + E @ (-2 * I + L * dt))
    coef3 = -4 * I - 3 * L * dt - L_sq * dt**2 + E @ (4 * I - L * dt)

    return E, E2, phi, L_inv3, coef1, coef2, coef3


def my_etdrk4(etdrk4_coefs, fun_nonlinear, t_vec, x0, internal_steps=1, args=()):
    """
    Integrates a system of ordinary differential equations using the Exponential Time Differencing Runge-Kutta 4 (ETDRK4) method.
    Assumes linear operator has been diagonalized and all necessary constant matrices have been precomputed from `etdrk4_setup`.
    See https://epubs.siam.org/doi/10.1137/S1064827502410633

    Batched support:
      - x0 can be shape (n,) or (B, n). In batched mode, fun_nonlinear should accept x with shape (B, n)
        and return the same shape.
    """

    dtype_real = x0.dtype
    dtype = torch.complex64 if dtype_real == torch.float32 else torch.complex128
    # Cast x0 and any tensor args to complex dtype
    x0 = x0.to(dtype)
    args = tuple(
        arg.to(dtype) if isinstance(arg, torch.Tensor) else arg for arg in args
    )

    # Normalize to batched shape (B, n)
    if x0.ndim == 1:
        batched = False
        n = x0.shape[0]
        x_b = x0.unsqueeze(0)  # (1, n)
    elif x0.ndim == 2:
        batched = True
        B, n = x0.shape
        x_b = x0  # (B, n)
    else:
        raise ValueError("x0 must be 1D (n,) or 2D (B, n).")

    dt = (t_vec[1] - t_vec[0]) / internal_steps
    E, E2, phi, L_inv3, coef1, coef2, coef3 = etdrk4_coefs

    n_outputs = len(t_vec)
    n_steps = (n_outputs - 1) * internal_steps
    t_vec_internal = (
        torch.arange(n_steps + 1, device=x_b.device, dtype=t_vec.dtype) * dt + t_vec[0]
    )

    xs_b = torch.zeros((x_b.shape[0], n, n_outputs), device=x_b.device, dtype=x_b.dtype)
    xs_b[:, :, 0] = x_b
    x = x_b.clone().detach()

    # Helper for left-multiplication A @ X for batched X: returns (B, n)
    # Implemented as (B, n) @ (n, n)^T to avoid expanding A to batches.
    def matvec(A, X):
        return X @ A.transpose(0, 1)

    for m in range(1, n_steps + 1):
        t = t_vec_internal[m - 1]
        N1 = fun_nonlinear(t, x, *args)
        an = matvec(E2, x) + matvec(phi, N1)
        N2 = fun_nonlinear(t + dt / 2, an, *args)
        bn = matvec(E2, x) + matvec(phi, N2)
        N3 = fun_nonlinear(t + dt / 2, bn, *args)
        cn = matvec(E2, an) + matvec(phi, (2 * N3 - N1))
        N4 = fun_nonlinear(t + dt, cn, *args)

        combo = matvec(coef1, N1) + matvec(coef2, (N2 + N3)) + matvec(coef3, N4)
        x = matvec(E, x) + (dt ** (-2)) * matvec(L_inv3, combo)

        if m % internal_steps == 0:
            out_idx = m // internal_steps
            xs_b[:, :, out_idx] = x.real

    # Match original return shape
    if batched:
        return xs_b.real
    else:
        return xs_b[0].real
