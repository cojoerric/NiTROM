import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpi4py import MPI

import time_steppers as tstep
import post_process as pp

from NiTROM.Optimization_Functions import classes
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

cPOD, cOI, cOI_gs, cNIT, cNIT_gs = '#66c2a5', '#fc8d62', '#fc8d62', '#8da0cb','#8da0cb'
lPOD, lOI, lOI_gs, lNIT, lNIT_gs = 'dotted', 'dashed', 'solid', 'dashed', 'solid'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

if rank == 0:
    print(f"Using {world_size} MPI process(es) for CPU analysis.")

##
Lx = 1
Ly = 1
Nx = 100
Ny = 100

dx = Lx/Nx
dy = Ly/Ny
Re = 8300

flow = classes_cavity.flow_class(Lx,Ly,Nx,Ny,Re)

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
phi_pre = np.load(traj_path + "phi_pre.npy")
n_traj = len(amps)
n = phi_pre.shape[-1]

pool_inputs = (comm, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

r = 50               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part

which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 2

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

phi_pod = np.eye(n, r)
psi_pod = phi_pod.copy()
A_pod = np.load('results/A_pod.npy')
H_pod = np.load('results/H_pod.npy')
tensors_pod = (A_pod, H_pod)

A_oi = np.load('results/A_oi.npy')
H_oi = np.load('results/H_oi.npy')
tensors_oi = (A_oi, H_oi)

A_oi_gs = np.load('results/A_oi_gs.npy')
H_oi_gs = np.load('results/H_oi_gs.npy')
tensors_oi_gs = (A_oi_gs, H_oi_gs)

phi_nit = np.load('results/phi_nit.npy')
psi_nit = np.load('results/psi_nit.npy')
A_nit = np.load('results/A_nit.npy')
H_nit = np.load('results/H_nit.npy')
tensors_nit = (A_nit, H_nit)

phi_nit_gs = np.load('results/phi_nit_gs.npy')
psi_nit_gs = np.load('results/psi_nit_gs.npy')
A_nit_gs = np.load('results/A_nit_gs.npy')
H_nit_gs = np.load('results/H_nit_gs.npy')
tensors_nit_gs = (A_nit_gs, H_nit_gs)


## Plot results
time = pool.time
u = np.zeros(r)
e_pod = 0
e_oi = 0
e_oi_gs = 0
e_nit = 0
e_nit_gs = 0

plt.figure()
for k in range(n_traj):
    mean_en = np.mean(np.linalg.norm(pool.X[k,], axis=0)**2)

    # POD
    z_pod = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_pod, method='RK45', t_eval=time, args=(u,) + tensors_pod).y
    e_pod = np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en
    if k == 0:
        plt.plot(time, e_pod, color=cPOD, linestyle=lPOD, alpha=1.0, label='POD-Gal.')
    else:
        plt.plot(time, e_pod, color=cPOD, linestyle=lPOD, alpha=0.3)

    # OpInf
    z_oi = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_oi, method='RK45', t_eval=time, args=(u,) + tensors_oi).y
    e_oi = np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en
    if k == 0:
        plt.plot(time, e_oi, color=cOI, linestyle=lOI, alpha=1.0, label='OpInf')
    else:
        plt.plot(time, e_oi, color=cOI, linestyle=lOI, alpha=0.3)

    # OpInf GS
    z_oi_gs = psi_pod.T @ pool.X[k,:,0]
    sol = phi_pod @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_oi_gs, method='RK45', t_eval=time, args=(u,) + tensors_oi_gs).y
    e_oi_gs = np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en
    if k == 0:
        plt.plot(time, e_oi_gs, color=cOI_gs, linestyle=lOI_gs, alpha=1.0, label='OpInf-GS')
    else:
        plt.plot(time, e_oi_gs, color=cOI_gs, linestyle=lOI_gs, alpha=0.3)

    # NiTROM
    z_nit = psi_nit.T @ pool.X[k,:,0]
    sol = phi_nit @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_nit, method='RK45', t_eval=time, args=(u,) + tensors_nit).y
    e_nit = np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en
    if k == 0:
        plt.plot(time, e_nit, color=cNIT, linestyle=lNIT, alpha=1.0, label='NiTROM')
    else:
        plt.plot(time, e_nit, color=cNIT, linestyle=lNIT, alpha=0.3)

    # NiTROM GS
    z_nit_gs = psi_nit_gs.T @ pool.X[k,:,0]
    sol = phi_nit_gs @ solve_ivp(opt_obj.evaluate_rom_rhs, [time[0], time[-1]], z_nit_gs, method='RK45', t_eval=time, args=(u,) + tensors_nit_gs).y
    e_nit_gs = np.linalg.norm(sol - pool.X[k,], axis=0)**2 / mean_en
    if k == 0:
        plt.plot(time, e_nit_gs, color=cNIT_gs, linestyle=lNIT_gs, alpha=1.0, label='NiTROM-GS')
    else:
        plt.plot(time, e_nit_gs, color=cNIT_gs, linestyle=lNIT_gs, alpha=0.3)


plt.xlabel('Time')
plt.ylabel('Error')
ax = plt.gca()
ax.set_yscale('log')
ax.set_ylim(bottom=1e-3)
plt.tight_layout()
plt.savefig('figures/errors')