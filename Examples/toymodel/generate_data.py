import dill
import numpy as np
import torch

import fom_class
from nitrom.time_steppers.time_stepper import solve_ivp


# %% Instantiate the full-order model class

device = "cpu"
dtype = torch.float64

n = 3
beta = 20.0
A2 = torch.diag(torch.tensor([-1.0, -2.0, -5.0], device=device, dtype=dtype))
A3 = torch.zeros((n, n, n), device=device, dtype=dtype)
A3[:, :, -1] = torch.diag(torch.tensor([beta, beta, 0.0], device=device, dtype=dtype))
B = torch.ones((n, 1), device=device, dtype=dtype)
C = torch.ones((1, n), device=device, dtype=dtype)
time = torch.linspace(0.0, 10.0, steps=100, device=device, dtype=dtype)

fom = fom_class.full_order_model(A2, A3, B, C, device=device, dtype=dtype)

# %% Generate training trajectories and save to file

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.pkl"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

betas = torch.tensor([0.01, 0.1, 0.2, 0.248], device=device, dtype=dtype)
n_traj = len(betas)

# Fixed-step RK4 sub-step used to integrate the FOM between saved snapshots.
dt = float((time[1] - time[0]) / 100)

for k in range(n_traj):
    b = float(betas[k])

    # Full-space forcing field f(t) = B u = b * ones(n) that drives the FOM.
    u = lambda t, b=b: b * torch.ones(n, device=device, dtype=dtype)

    # Integrate dx/dt = A2 x + A3 : x x^T + f(t) with the torch RK4 stepper.
    X = solve_ivp(
        fom.evaluate_fom_dynamics,
        torch.zeros(n, device=device, dtype=dtype),
        float(time[0]),
        float(time[-1]),
        dt,
        time,
        "rk4",
        u,
    )  # (n, n_time)

    # Time derivatives at the saved snapshots.
    dX = torch.zeros_like(X)
    for j in range(time.shape[0]):
        dX[:, j] = fom.evaluate_fom_dynamics(float(time[j]), X[:, j], u)

    # Analytic steady state -> trajectory weight = ||C x_ss||^2.
    id_ss = torch.tensor(
        [
            -b / (-1 + beta / 5 * b),
            -b / (-2 + beta / 5 * b),
            b / 5,
        ],
        device=device,
        dtype=dtype,
    )
    weight = torch.linalg.vector_norm(fom.compute_output(id_ss)) ** 2

    # Saved forcing is the m = 1 ROM input u(t) = b (multiplied by B_r downstream).
    u_save = lambda t, b=b: b * torch.ones(1)

    np.save(fname_traj % k, X.cpu().numpy())
    np.save(fname_deriv % k, dX.cpu().numpy())
    np.save(fname_weight % k, np.array([float(weight)]))
    with open(fname_forcing % k, "wb") as f:
        dill.dump(u_save, f)

np.save(fname_time, time.cpu().numpy())
