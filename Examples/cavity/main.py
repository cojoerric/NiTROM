import torch
import numpy as np
import matplotlib.pyplot as plt

import time_steppers as tstep
import post_process as pp

from NiTROM.Optimization_Functions import classes, nitrom_models as nit_model, opinf_models as oi_model, opinf_closed_form as oi_cf, utils
from NiTROM.PyTorch_Functions import gpu_utils, train
import classes_cavity

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.fontsize"] = 14
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2
torch.set_printoptions(precision=8)


device, rank, world_size = gpu_utils.setup_distributed_gpus()
dtype = torch.float64
if rank == 0:
    print(f"Using {world_size} GPU(s) for distributed training.")
    print(f"Device: {device}")

Lx = 1
Ly = 1
Nx = 100
Ny = 100

dx = Lx/Nx
dy = Ly/Ny
Re = 8300

flow = classes_cavity.flow_parameters(Lx,Ly,Nx,Ny,Re)

n = 400
dt = 1.0/n

lops = classes_cavity.linear_operators_2D(flow,dt)
flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))
fom = classes_cavity.fom_class(flow,lops)
fom.assemble_forcing_profile(0.95, 0.05)

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

amps = np.load(traj_path + "amps.npy")
phi_pre = np.load(traj_path + "Phi_pre.npy")
n = phi_pre.shape[-1]

n_traj = len(amps)
pool_inputs = (n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
               'dtype':dtype,
               'device':device,
               'rank':rank,
               'world_size':world_size
}
pool = classes.pool(*pool_inputs,**pool_kwargs)

r = 50               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part

which_trajs = torch.arange(0,pool.n_traj,1,device=device)
which_times = torch.arange(0,pool.n_snapshots,1,device=device)
leggauss_deg = 5
nsave_rom = 2

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

phi_pod = np.eye(n, r)
psi_pod = phi_pod.copy()
phi_tot = phi_pre @ phi_pod
psi_tot = phi_pre @ psi_pod
phi_pod = torch.tensor(phi_pod, device=device, dtype=dtype)
psi_pod = torch.tensor(psi_pod, device=device, dtype=dtype)


## Compute POD model
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(phi_tot, psi_tot)
np.save('results/A_pod.npy', tensors_pod[0])
np.save('results/H_pod.npy', tensors_pod[1])
tensors_pod = tuple([torch.tensor(tensor, device=device, dtype=dtype) for tensor in tensors_pod])
A_pod, H_pod = tensors_pod


## Compute OpInf model
print("\nComputing OpInf Model...")
tensors_oi = oi_cf.operator_inference(pool, phi_pod, poly_comp, lambdas=[0.0, 0.0])
A_oi, H_oi = tensors_oi

np.save('results/A_oi.npy', A_oi.cpu().numpy())
np.save('results/H_oi.npy', H_oi.cpu().numpy())


## Compute globally stable OpInf model
print("\nTraining OpInf (GS) Model...")
init = {
    "A2": A_pod,
    "A3": H_pod,
}
params_oi = oi_model.OpinfParams_GloballyStable(pool, r, poly_comp, init=init, requires_grad=True)
model_oi = oi_model.OpinfModel(phi_pod, params_oi, opt_obj).to(device)
optimizer_oi = torch.optim.LBFGS(model_oi.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn='strong_wolfe')

model_oi, history = train.train_model(
    model_oi,
    pool,
    optimizer_oi,
    num_epochs=500,
    log_every=50,
)

Qhat = model_oi.params.Qhat.detach()
Jhat = model_oi.params.Jhat.detach()
Rhat = model_oi.params.Rhat.detach()
Hhat = model_oi.params.Hhat.detach()
A_oi_gs, H_oi_gs = utils.construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]
tensors_oi_gs = (A_oi_gs, H_oi_gs)

np.save('results/A_oi_gs.npy', A_oi_gs.cpu().numpy())
np.save('results/H_oi_gs.npy', H_oi_gs.cpu().numpy())


## Compute NiTROM model
print("\nTraining NiTROM Model...")
init = {
    "Phi": phi_pod,
    "Psi": psi_pod,
    "A2": A_pod,
    "A3": H_pod,
}
params_nit = nit_model.NitromParams(pool, r, poly_comp, init=init, requires_grad=True)
model_nit = nit_model.NitromModel(params_nit, opt_obj, fom).to(device)
optimizer_nit = torch.optim.LBFGS(model_nit.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn='strong_wolfe')

model_nit, history = train.train_model(
    model_nit,
    pool,
    optimizer_nit,
    num_epochs=50,
    log_every=1,
    manifold_retraction="qr",
)

phi_nit = model_nit.params.Phi.detach()
psi_nit = model_nit.params.Psi.detach()
A_nit = model_nit.params.A2.detach()
H_nit = model_nit.params.A3.detach()
tensors_nit = (A_nit, H_nit)

np.save('results/phi_nit.npy', phi_nit.cpu().numpy())
np.save('results/psi_nit.npy', psi_nit.cpu().numpy())
np.save('results/A_nit.npy', A_nit.cpu().numpy())
np.save('results/H_nit.npy', H_nit.cpu().numpy())


## Compute globally stable NiTROM model
print("\nTraining NiTROM (GS) Model...")
init = utils.create_intitial_guess(A_pod, H_pod, r=r)
init["Phi"] = phi_pod.clone()
init["Psi"] = psi_pod.clone()

params_nit_gs = nit_model.NitromParams_GloballyStable(pool, r, poly_comp, init=init, requires_grad=True)
model_nit_gs = nit_model.NitromModel_GloballyStable(params_nit_gs, opt_obj, fom).to(device)
optimizer_nit_gs = torch.optim.LBFGS(model_nit_gs.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn='strong_wolfe')

model_nit_gs, history = train.train_model(
    model_nit_gs,
    pool,
    optimizer_nit_gs,
    num_epochs=50,
    log_every=1,
    manifold_retraction="qr",
)

phi_nit_gs = model_nit_gs.params.Phi.detach()
psi_nit_gs = model_nit_gs.params.Psi.detach()
Qhat = model_nit_gs.params.Qhat.detach()
Jhat = model_nit_gs.params.Jhat.detach()
Rhat = model_nit_gs.params.Rhat.detach()
Hhat = model_nit_gs.params.Hhat.detach()
A_nit_gs, H_nit_gs = utils.construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]

np.save('results/phi_nit_gs.npy', phi_nit_gs.cpu().numpy())
np.save('results/psi_nit_gs.npy', psi_nit_gs.cpu().numpy())
np.save('results/A_nit_gs.npy', A_nit_gs.cpu().numpy())
np.save('results/H_nit_gs.npy', H_nit_gs.cpu().numpy())