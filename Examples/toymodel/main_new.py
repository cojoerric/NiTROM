import torch
import numpy as np
import matplotlib.pyplot as plt

import nitrom

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16
plt.rcParams["lines.linewidth"] = 2
torch.set_printoptions(precision=8)

cPOD, cOI, cOI_gs, cNIT, cNIT_gs = "#66c2a5", "#fc8d62", "#8da0cb", "#fc8d62", "#8da0cb"
lPOD, lOI, lOI_gs, lNIT, lNIT_gs = "solid", "dotted", "dotted", "dashed", "dashed"

device, rank, world_size = nitrom.setup_distributed()
dtype = torch.float64
if rank == 0:
    print(f"Using {world_size} devices for distributed training.")
    print(f"Device: {device}")

n = 3
n_traj = 4
beta = 20.0
diag_vec = torch.tensor([-1.0, -2.0, -5.0], device=device, dtype=dtype)
A2 = torch.diag(diag_vec)
A3 = torch.zeros((3, 3, 3), device=device, dtype=dtype)
diag_vec2 = torch.tensor([beta, beta, 0.0], device=device, dtype=dtype)
A3[:, :, -1] = torch.diag(diag_vec2)
B = torch.ones((3, 1), device=device, dtype=dtype)
C = torch.ones((1, 3), device=device, dtype=dtype)

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.pkl"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

pool_inputs = (n_traj, fname_traj, fname_time, dtype, device, rank, world_size)
pool_kwargs = {
    "fname_forcing": fname_forcing,
    "fname_weights": fname_weight,
    "fname_derivs": fname_deriv,
}

pool = nitrom.TrainingPool(*pool_inputs, **pool_kwargs)
opt_obj = nitrom.TrainingData(
    pool,
    which_trajs=[0, 1],
    percent_time_length=0.5,
    leggauss_deg=3,
    nsave_rom=10,
)

Phi = nitrom.perform_POD(pool, 2)
poly_comp = [1, 2]
opinf_class = nitrom.OpInfCostAndGrad(opt_obj, poly_comp, Phi, 0, True)
trained_params = nitrom.train_opinf(
    opinf_class,
    n_epochs=10000,
    lr=1e-3,
    optimizer_type="lbfgs",
    print_every=1,
    tol=1e-12,
)

nitrom.cleanup_distributed()
