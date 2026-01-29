import torch
import numpy as np
import matplotlib.pyplot as plt

import time_steppers as tstep
import post_process as pp

from NiTROM.Optimization_Functions import classes
from NiTROM.PyTorch_Functions import gpu_utils
from NiTROM.PyTorch_Functions.integrators import my_rk4_adaptive
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


cPOD, cOI, cOI_gs, cNIT, cNIT_gs = '#66c2a5', '#fc8d62', '#8da0cb', '#fc8d62','#8da0cb'
lPOD, lOI, lOI_gs, lNIT, lNIT_gs = 'solid', 'dotted', 'dotted', 'dashed', 'dashed'

device, rank, world_size = gpu_utils.setup_distributed_gpus()
dtype = torch.float64
if rank == 0:
    print(f"Using {world_size} GPU(s) for distributed training.")
    print(f"Device: {device}")

##
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


##
traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

amps = np.load(traj_path + "amps.npy")
phi_pre = np.load(traj_path + "Phi_pre.npy")
n_traj = len(amps)
n = phi_pre.shape[-1]

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

phi_pod = torch.eye(n, r, device=device, dtype=dtype)
psi_pod = phi_pod.clone()
A_pod = torch.tensor(np.load('results/A_pod.npy'), device=device, dtype=dtype)
H_pod = torch.tensor(np.load('results/H_pod.npy'), device=device, dtype=dtype)
tensors_pod = (A_pod, H_pod)

A_oi = torch.tensor(np.load('results/A_oi.npy'), device=device, dtype=dtype)
H_oi = torch.tensor(np.load('results/H_oi.npy'), device=device, dtype=dtype)
tensors_oi = (A_oi, H_oi)

A_oi_gs = torch.tensor(np.load('results/A_oi_gs.npy'), device=device, dtype=dtype)
H_oi_gs = torch.tensor(np.load('results/H_oi_gs.npy'), device=device, dtype=dtype)
tensors_oi_gs = (A_oi_gs, H_oi_gs)

phi_nit = torch.tensor(np.load('results/phi_nit.npy'), device=device, dtype=dtype)
psi_nit = torch.tensor(np.load('results/psi_nit.npy'), device=device, dtype=dtype)
A_nit = torch.tensor(np.load('results/A_nit.npy'), device=device, dtype=dtype)
H_nit = torch.tensor(np.load('results/H_nit.npy'), device=device, dtype=dtype)
tensors_nit = (A_nit, H_nit)

phi_nit_gs = torch.tensor(np.load('results/phi_nit_gs.npy'), device=device, dtype=dtype)
psi_nit_gs = torch.tensor(np.load('results/psi_nit_gs.npy'), device=device, dtype=dtype)
A_nit_gs = torch.tensor(np.load('results/A_nit_gs.npy'), device=device, dtype=dtype)
H_nit_gs = torch.tensor(np.load('results/H_nit_gs.npy'), device=device, dtype=dtype)
tensors_nit_gs = (A_nit_gs, H_nit_gs)


## Plot results
time = pool.time.cpu().numpy()
u = torch.zeros(r, device=device, dtype=dtype)
e_pod = 0
e_oi = 0
e_oi_gs = 0
e_nit = 0
e_nit_gs = 0

plt.figure()
for k in range(n_traj):
    mean_en = torch.mean(torch.linalg.norm(pool.X[k,], dim=0)**2)

    # POD
    z_pod = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_pod, args=(u,) + tensors_pod)
    e_pod += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # e_pod = torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en
    # plt.plot(time, e_pod.cpu().numpy(), color=cPOD, linestyle=lPOD, alpha=0.3)

    # OpInf
    z_oi = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_oi, args=(u,) + tensors_oi)
    e_oi += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # plt.plot(time, torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en, color=cOI, linestyle=lOI, alpha=0.3)

    # OpInf GS
    z_oi_gs = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_oi_gs, args=(u,) + tensors_oi_gs)
    e_oi_gs += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # e_oi_gs = torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en
    # plt.plot(time, e_oi_gs.cpu().numpy(), color=cOI_gs, linestyle=lOI_gs, alpha=0.3)

    # NiTROM
    z_nit = psi_nit.T @ pool.X[k,:,0]
    sol = phi_nit @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_nit, args=(u,) + tensors_nit)
    e_nit += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # plt.plot(time, torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en, color=cNIT, linestyle=lNIT, alpha=0.3)

    # NiTROM GS
    z_nit_gs = psi_nit_gs.T @ pool.X[k,:,0]
    sol = phi_nit_gs @ my_rk4_adaptive(opt_obj.evaluate_rom_rhs, pool.time, z_nit_gs, args=(u,) + tensors_nit_gs)
    e_nit_gs += torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en / n_traj
    # e_nit_gs = torch.linalg.norm(sol - pool.X[k,], dim=0)**2 / mean_en
    # plt.plot(time, e_nit_gs.cpu().numpy(), color=cNIT_gs, linestyle=lNIT_gs, alpha=0.3)


plt.plot(time, e_pod.cpu().numpy(), label='POD', color=cPOD, linestyle=lPOD)
plt.plot(time, e_oi.cpu().numpy(), label='OpInf', color=cOI, linestyle=lOI)
plt.plot(time, e_oi_gs.cpu().numpy(), label='OpInf (GS)', color=cOI_gs, linestyle=lOI_gs)
plt.plot(time, e_nit.cpu().numpy(), label='NiTROM', color=cNIT, linestyle=lNIT)
plt.plot(time, e_nit_gs.cpu().numpy(), label='NiTROM (GS)', color=cNIT_gs, linestyle=lNIT_gs)
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
ax = plt.gca()
ax.set_yscale('log')
ax.set_ylim(bottom=1e-3)
plt.tight_layout()
plt.savefig('errors')