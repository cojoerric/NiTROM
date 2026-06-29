import torch
import numpy as np
import matplotlib.pyplot as plt

from NiTROM.Optimization_Functions import (
    classes,
    nitrom_models as nit_model,
    opinf_models as oi_model,
    opinf_closed_form as oi_cf,
    utils,
)
from NiTROM.PyTorch_Functions import gpu_utils, train, integrators
from NiTROM.PyTorch_Functions.linear_interpolation import Interp1D
import fom_class_pytorch

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

device, rank, world_size = gpu_utils.setup_distributed_gpus()
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
fom = fom_class_pytorch.full_order_model(A2, A3, B, C, device=device, dtype=dtype)

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

pool_inputs = (n_traj, fname_traj, fname_time)
pool_kwargs = {
    "fname_steady_forcing": fname_forcing,
    "fname_weights": fname_weight,
    "fname_derivs": fname_deriv,
    "dtype": dtype,
    "device": device,
    "rank": rank,
    "world_size": world_size,
}
pool = classes.pool(*pool_inputs, **pool_kwargs)

r = 2  # ROM dimension
poly_comp = [1, 2]  # Model with a linear part and a quadratic part

# %% Compute NiTROM model

which_trajs = torch.arange(0, pool.n_traj, 1, device=device)
which_times = torch.arange(0, pool.n_snapshots, 1, device=device)
leggauss_deg = 5
nsave_rom = 2
integrator = integrators.my_rk4_adaptive

opt_obj_inputs = (pool, which_trajs, which_times, leggauss_deg, nsave_rom, poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

N = pool.n_snapshots * pool.n_traj
X = torch.zeros((pool.X.shape[1], N), device=device, dtype=dtype)
for i in range(pool.n_traj):
    X[:, i * pool.n_snapshots : (i + 1) * pool.n_snapshots] = pool.X[i,]
phi_pod, _, _ = torch.linalg.svd(X, full_matrices=False)
phi_pod = phi_pod[:, :r]
psi_pod = phi_pod.clone()

## Compute POD model
print("\nComputing POD Model...")
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(phi_pod, psi_pod)
np.save("results/A_pod.npy", tensors_pod[0].cpu().numpy())
np.save("results/H_pod.npy", tensors_pod[1].cpu().numpy())
tensors_pod = tuple(
    [torch.tensor(tensor, device=device, dtype=dtype) for tensor in tensors_pod]
)
A_pod, H_pod = tensors_pod


## Compute OpInf model
print("\nComputing OpInf Model...")
weights = pool.weights.clone()
pool.weights *= pool.n_traj * pool.n_snapshots
lambdas = torch.logspace(-8, -2, steps=100, device=device, dtype=dtype)
# lambdas = torch.tensor([0.0], device=device, dtype=dtype)
cost_oi = torch.zeros_like(lambdas)
for i, lamb in enumerate(lambdas):
    print(f"Lambda: {lamb:.2e}")
    tensors_oi = oi_cf.operator_inference(pool, phi_pod, poly_comp, lambdas=[0.0, lamb])
    A_oi, H_oi = tensors_oi
    point = {
        "Phi": phi_pod,
        "Psi": psi_pod,
        "A2": A_oi,
        "A3": H_oi,
    }
    params_nit = nit_model.NitromParams(
        pool, r, poly_comp, init=point, requires_grad=True
    ).to(device)
    model_nit = nit_model.NitromModel(params_nit, opt_obj, fom, integrator).to(device)
    cost_oi[i] = model_nit().item()

pool.weights = weights
lambdas_idx = torch.argmin(cost_oi)
lambdas = lambdas[lambdas_idx]
print(torch.min(cost_oi), lambdas)
weights = pool.weights.clone()
pool.weights *= pool.n_traj * pool.n_snapshots
tensors_oi = oi_cf.operator_inference(pool, phi_pod, poly_comp, lambdas=[0.0, lambdas])
A_oi, H_oi = tensors_oi
pool.weights = weights

np.save("results/A_oi.npy", A_oi.cpu().numpy())
np.save("results/H_oi.npy", H_oi.cpu().numpy())


## Compute globally stable OpInf model
print("\nTraining OpInf (GS) Model...")
weights = pool.weights.clone()
pool.weights *= pool.n_traj * pool.n_snapshots
lambdas = torch.logspace(-8, -2, steps=100, device=device, dtype=dtype)
# lambdas = torch.tensor([0.0], device=device, dtype=dtype)
cost_oi_gs = torch.zeros_like(lambdas)
init = utils.create_initial_guess(A_pod, H_pod, r=r)
for i, lamb in enumerate(lambdas):
    print(f"Lambda: {lamb:.2e}")
    params_oi = oi_model.OpinfParams_GloballyStable(
        pool, r, poly_comp, init=init, requires_grad=True
    ).to(device)
    model_oi = oi_model.OpinfModel(
        phi_pod, params_oi, opt_obj, regularization_H=lamb
    ).to(device)
    optimizer_oi = torch.optim.LBFGS(
        model_oi.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    model_oi, history = train.train_model(
        model_oi,
        pool,
        optimizer_oi,
        num_epochs=60,
        log_every=30,
    )
    Qhat = model_oi.params.Qhat.detach()
    Jhat = model_oi.params.Jhat.detach()
    Rhat = model_oi.params.Rhat.detach()
    Hhat = model_oi.params.Hhat.detach()
    point = {
        "Phi": phi_pod,
        "Psi": psi_pod,
        "Qhat": Qhat,
        "Jhat": Jhat,
        "Rhat": Rhat,
        "Hhat": Hhat,
    }
    params_nit_gs = nit_model.NitromParams_GloballyStable(
        pool, r, poly_comp, init=point, requires_grad=True
    ).to(device)
    model_nit_gs = nit_model.NitromModel(params_nit_gs, opt_obj, fom, integrator).to(
        device
    )
    cost_oi_gs[i] = model_nit_gs().item()

pool.weights = weights
lambdas_idx = torch.argmin(cost_oi_gs)
lambdas = lambdas[lambdas_idx]
print(torch.min(cost_oi_gs), lambdas)
weights = pool.weights.clone()
pool.weights *= pool.n_traj * pool.n_snapshots
params_oi = oi_model.OpinfParams_GloballyStable(
    pool, r, poly_comp, init=init, requires_grad=True
).to(device)
model_oi = oi_model.OpinfModel(
    phi_pod, params_oi, opt_obj, regularization_H=lambdas
).to(device)
optimizer_oi = torch.optim.LBFGS(
    model_oi.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=10,
    line_search_fn="strong_wolfe",
)
model_oi, history = train.train_model(
    model_oi,
    pool,
    optimizer_oi,
    num_epochs=60,
    log_every=30,
)
Qhat = model_oi.params.Qhat.detach()
Jhat = model_oi.params.Jhat.detach()
Rhat = model_oi.params.Rhat.detach()
Hhat = model_oi.params.Hhat.detach()
A_oi_gs, H_oi_gs = utils.construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]
tensors_oi_gs = (A_oi_gs, H_oi_gs)
pool.weights = weights

np.save("results/A_oi_gs.npy", A_oi_gs.cpu().numpy())
np.save("results/H_oi_gs.npy", H_oi_gs.cpu().numpy())


## Compute NiTROM model
print("\nTraining NiTROM Model...")
init = {
    "Phi": phi_pod,
    "Psi": psi_pod,
    "A2": A_pod,
    "A3": H_pod,
}
params_nit = nit_model.NitromParams(
    pool, r, poly_comp, init=init, requires_grad=True
).to(device)
model_nit = nit_model.NitromModel(params_nit, opt_obj, fom, integrator).to(device)
optimizer_nit = torch.optim.LBFGS(
    model_nit.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=10,
    line_search_fn="strong_wolfe",
)

model_nit, history = train.train_model(
    model_nit,
    pool,
    optimizer_nit,
    num_epochs=20,
    log_every=1,
    manifold_retraction="qr",
    vector_transport=True,
)

phi_nit = model_nit.params.Phi.detach()
psi_nit = model_nit.params.Psi.detach()
A_nit = model_nit.params.A2.detach()
H_nit = model_nit.params.A3.detach()
tensors_nit = (A_nit, H_nit)

np.save("results/phi_nit.npy", phi_nit.cpu().numpy())
np.save("results/psi_nit.npy", psi_nit.cpu().numpy())
np.save("results/A_nit.npy", A_nit.cpu().numpy())
np.save("results/H_nit.npy", H_nit.cpu().numpy())


## Compute globally stable NiTROM model
print("\nTraining NiTROM (GS) Model...")
# init = utils.create_initial_guess(A_pod, H_pod, r=r)
# init["Phi"] = phi_pod.clone()
# init["Psi"] = psi_pod.clone()
init = {
    "Phi": phi_pod,
    "Psi": psi_pod,
    "Jhat": Jhat,
    "Qhat": Qhat,
    "Rhat": Rhat,
    "Hhat": Hhat,
}

params_nit_gs = nit_model.NitromParams_GloballyStable(
    pool, r, poly_comp, init=init, requires_grad=True
).to(device)
model_nit_gs = nit_model.NitromModel(params_nit_gs, opt_obj, fom, integrator).to(device)
optimizer_nit_gs = torch.optim.LBFGS(
    model_nit_gs.parameters(),
    lr=1.0,
    max_iter=20,
    history_size=10,
    line_search_fn="strong_wolfe",
)

model_nit_gs, history = train.train_model(
    model_nit_gs,
    pool,
    optimizer_nit_gs,
    num_epochs=15,
    log_every=1,
    manifold_retraction="qr",
    vector_transport=True,
)

phi_nit_gs = model_nit_gs.params.Phi.detach()
psi_nit_gs = model_nit_gs.params.Psi.detach()
Qhat = model_nit_gs.params.Qhat.detach()
Jhat = model_nit_gs.params.Jhat.detach()
Rhat = model_nit_gs.params.Rhat.detach()
Hhat = model_nit_gs.params.Hhat.detach()
A_nit_gs, H_nit_gs = utils.construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]
tensors_nit_gs = (A_nit_gs, H_nit_gs)

np.save("results/phi_nit_gs.npy", phi_nit_gs.cpu().numpy())
np.save("results/psi_nit_gs.npy", psi_nit_gs.cpu().numpy())
np.save("results/A_nit_gs.npy", A_nit_gs.cpu().numpy())
np.save("results/H_nit_gs.npy", H_nit_gs.cpu().numpy())
