import torch

def rk4_step(fun, t, x, dt, args=()):
    """
    Performs a single integration step using the fourth-order Runge-Kutta (RK4) method.
    """
    
    k1 = fun(t, x, *args)
    k2 = fun(t + 0.5 * dt, x + 0.5 * dt * k1, *args)
    k3 = fun(t + 0.5 * dt, x + 0.5 * dt * k2, *args)
    k4 = fun(t + dt, x + dt * k3, *args)
    x_new = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_new

@torch.compile
def myRK4(fun, t_vec, x0, args=(), *, atol=1e-6, rtol=1e-3, safety_factor=0.8, fac_min=0.1, fac_max=5.0):
    """
    Integrates a system of ordinary differential equations using an adaptive fourth-order Runge-Kutta (RK4) method.
    Compatible with pyTorch tensors.

    Args:
        fun (callable): The function that computes the derivatives, with signature `fun(t, x, *args)`.
        t_vec (torch.Tensor): 1D tensor or array of time points at which to solve for the state.
        x0 (torch.Tensor): Initial state vector.
        args (tuple, optional): Additional arguments to pass to `fun`.

    Returns:
        xs: A tensor containing the state at each time point in `t_vec`. Each column corresponds to the state at a time step.
    """

    x = x0.clone().detach()
    dt = (t_vec[1] - t_vec[0])/100
    xs = torch.zeros((len(x0), len(t_vec)), dtype=x0.dtype, device=x0.device)
    xs[:, 0] = x0
    t = t_vec[0]
    for i, T in enumerate(t_vec[1:], start=1):
        while t < T:
            dt_trial = min(dt, T - t)
            x_full = rk4_step(fun, t, x, dt_trial, args)
            x_half1 = rk4_step(fun, t, x, dt_trial / 2, args)
            x_half2 = rk4_step(fun, t + dt_trial / 2, x_half1, dt_trial / 2, args)

            dx = x_half2 - x_full
            scale = atol + rtol * torch.max(torch.abs(x_full), torch.abs(x_half2))
            err_vec = torch.abs(dx)/scale
            err = torch.max(err_vec)

            if err <= 1.0:
                t = t + dt_trial
                x = x_half2
            
            exponent = 1.0/(4.0+1.0)
            dt_new = dt_trial * safety_factor * (1.0/err)**exponent
            dt = torch.clamp(dt_new, min=dt*fac_min, max=dt*fac_max)
        xs[:, i] = x

    return xs


@torch.compile
def my_cnab2(linop, fun_nonlinear, t_vec, x0, args=()):
    dt = t_vec[1] - t_vec[0]
    dt_half = 0.5 * dt
    x_curr = x0.clone().detach()
    t_curr = t_vec[0]
    N_prev = fun_nonlinear(t_curr, x_curr, *args)

    I = torch.eye(len(x0), dtype=x0.dtype, device=x0.device)
    A = I - dt_half * linop
    B = I + dt_half * linop
    LU, pivots, _ = torch.linalg.lu_factor_ex(A)

    xs = torch.zeros((len(x0), len(t_vec)), dtype=x0.dtype, device=x0.device)
    xs[:, 0] = x_curr

    for i, t_next in enumerate(t_vec[1:], start=1):
        N_curr = fun_nonlinear(t_curr, x_curr, *args)
        rhs = B @ x_curr + dt_half * (3 * N_curr - N_prev)
        rhs = rhs.unsqueeze(-1)
        x_next = torch.linalg.lu_solve(LU, pivots, rhs).squeeze(-1)
        xs[:, i] = x_next

        t_curr = t_next
        x_curr = x_next
        N_prev = N_curr

    return xs