import os

import matplotlib.pyplot as plt
import torch

from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.plotting import COLORS, set_plot_style
from nitrom.time_steppers.time_stepper import solve_ivp

set_plot_style()

device = "cpu"
dtype = torch.float64
models_dir = "./models/"
n = 3  # FOM dimension

# %% Full-order model operators (same toy model as generate_data.py)

beta = 20.0
A2 = torch.diag(torch.tensor([-1.0, -2.0, -5.0], device=device, dtype=dtype))
A3 = torch.zeros((n, n, n), device=device, dtype=dtype)
A3[:, :, -1] = torch.diag(torch.tensor([beta, beta, 0.0], device=device, dtype=dtype))
B = torch.ones((n, 1), device=device, dtype=dtype)
C = torch.ones((1, n), device=device, dtype=dtype)  # output operator y = C x


def fom_rhs(t, x, f):
    """Step-forced FOM dynamics, batched over the leading axis (f = B u)."""
    return x @ A2.T + torch.einsum("ijk,...j,...k->...i", A3, x, x) + f


# %% Load the trained ROMs (each is a PolynomialModel plus its POD basis Phi)


def load_rom(fname):
    ckpt = torch.load(models_dir + fname, weights_only=False)
    rom = PolynomialModel(
        ckpt["r"],
        ckpt["poly_comp"],
        device=device,
        dtype=dtype,
        tensors=ckpt["tensors"],
        forcing_config=ckpt["forcing_config"],
    )
    return rom, ckpt["Phi"].to(device=device, dtype=dtype)


models = {
    r"POD-Galerkin": (load_rom("galerkin_model.pt"), COLORS["galerkin"], "solid"),
    r"OpInf": (load_rom("opinf_model.pt"), COLORS["opinf"], "dotted"),
    r"GAS-OpInf": (load_rom("gas_opinf_model.pt"), COLORS["gas"], "dashed"),
}

# %% Random step-response trajectories (constant forcing u = b), batched.
# Initial condition is x(0) = 0, as in the training data (generate_data.py);
# each trajectory is driven by a constant input of amplitude b_j.

time = torch.linspace(0.0, 10.0, 200, device=device, dtype=dtype)
t0, tf = float(time[0]), float(time[-1])
dt_sub = float((time[1] - time[0]) / 5)  # 5 RK4 sub-steps per snapshot

n_test = 100
max_amplitude = 0.24
torch.manual_seed(0)
amplitudes = max_amplitude * torch.rand(n_test, device=device, dtype=dtype)

x0 = torch.zeros(n_test, n, device=device, dtype=dtype)
# FOM forcing field B u = b * ones(n) per trajectory (constant in time).
forcing_field = amplitudes[:, None] * B[:, 0]  # (n_test, n)
# ROM forcing: one callable per trajectory returning the m=1 input u(t) = b.
forcing_fns = [
    (lambda t, b=float(b): b * torch.ones(1, device=device, dtype=dtype))
    for b in amplitudes
]

X_fom = solve_ivp(fom_rhs, x0, t0, tf, dt_sub, time, "rk4", forcing_field)
Y_fom = torch.einsum("on,bnt->bot", C, X_fom)  # FOM output (n_test, n_out, nt)

# Per-trajectory weight alpha_j = ||C x_ss(b_j)||^2, the steady-state output
# energy (same definition as the training weights in generate_data.py).
ratio = beta / 5 * amplitudes
x_ss = torch.stack(
    [-amplitudes / (-1 + ratio), -amplitudes / (-2 + ratio), amplitudes / 5],
    dim=1,
)  # (n_test, n)
alpha = torch.linalg.vector_norm(x_ss @ C.T, dim=1) ** 2  # (n_test,)

# e(t) = (1/N) sum_j ||y_j(t) - yhat_j(t)||^2 / alpha_j
avg_error = {}
for name, ((rom, Phi), _color, _style) in models.items():
    z0 = torch.zeros(n_test, Phi.shape[1], device=device, dtype=dtype)
    Z_rom = solve_ivp(
        rom.evaluate_rhs, z0, t0, tf, dt_sub, time, "rk4",
        external_forcing=forcing_fns,
    )
    X_rom = torch.einsum("nr,brt->bnt", Phi, Z_rom)  # (n_test, n, nt)
    Y_rom = torch.einsum("on,bnt->bot", C, X_rom)  # ROM output (n_test, n_out, nt)
    sq_err = torch.linalg.vector_norm(Y_fom - Y_rom, dim=1) ** 2  # (n_test, nt)
    avg_error[name] = (sq_err / alpha[:, None]).mean(dim=0)  # (nt,)

# %% Plot the trajectory-averaged error over time

fig, ax = plt.subplots(figsize=(6.0, 4.0))
for name, (_, color, style) in models.items():
    ax.semilogy(
        time.cpu().numpy(),
        avg_error[name].cpu().numpy(),
        color=color,
        linestyle=style,
        label=name,
    )
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"Average error $e(t)$")
ax.set_title(rf"Step response (averaged over {n_test} trajectories)")
ax.set_xlim(time[0], time[-1])
ax.set_ylim(top=1e-1)
ax.legend()

figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)
out_path = os.path.join(figures_dir, "rom_step_error.pdf")
fig.savefig(out_path)
print(f"saved -> {out_path}")
