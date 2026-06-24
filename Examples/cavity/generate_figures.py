#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 19:00:43 2024

@author: alberto
"""

import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI
import time_steppers as tstep
import post_process as pp

from scipy.integrate import solve_ivp
import sys




plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')

sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")


import classes
import classes_cavity 


cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'

#%%

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
fom.assemble_forcing_profile(0.95,0.05)
B = fom.f.copy()

#%% Plot steady state solution

X, Y, fields = pp.output_fields(flow,flow.q_sbf)

color_map = plt.cm.get_cmap('bwr')

idx = 2
vmin = np.min(fields[idx]) 
vmax = -vmin

plt.figure()
plt.contourf(X[idx],Y[idx],np.flipud(fields[idx]),levels=100,cmap=color_map,vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar()

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.tight_layout()
# plt.savefig("./Figures/bflow_Re%d.eps"%Re,format='eps')


#%% Load train/testing trajectories
traj_path = "./trajectories/"
Phi_pre = np.load(traj_path + "Phi_pre.npy")

r = 50
n = Phi_pre.shape[-1]
Phi_pod = np.zeros((n,r))
Phi_pod[:r,:r] = np.eye(r)
Psi_pod = Phi_pod.copy()


which = 'test'
if which == 'train':



    fname_traj = traj_path + "traj_%03d.npy"
    fname_weight = traj_path + "weight_%03d.npy"
    fname_deriv = traj_path + "deriv_%03d.npy"
    fname_time = traj_path + "time.npy"
    amps = np.load(traj_path + "amps_train.npy")
    
else:
    
    fname_traj = traj_path + "test_traj_%03d.npy"
    fname_weight = traj_path + "test_weight_%03d.npy"
    fname_deriv = traj_path + "test_deriv_%03d.npy"
    fname_time = traj_path + "test_time.npy"
    amps = np.load(traj_path + "amps_test.npy")
    
    

n_traj = len(amps)


pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)
poly_comp = [1,2]   # Model with a linear part and a quadratic part

which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 15
opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,[1,2])
opt_obj = classes.optimization_objects(*opt_obj_inputs)

#%% Plot time-history of perturbation energy

plt.figure()
for k in range (pool.my_n_traj):
    Qk = pool.X[k,]
    energy_k = np.linalg.norm(Qk,axis=0)**2
    plt.plot(pool.time,energy_k,'k')
    

ax = plt.gca()
ax.set_xlabel('Time $t$')
ax.set_ylabel('Energy of perturbations')

plt.tight_layout()
# plt.savefig("./Figures/energy_perturbations.eps",format='eps')


#%%

Phi_nit = np.load("data/Phi_nit.npy")
Psi_nit = np.load("data/Psi_nit.npy")
A2_nit = np.load("data/A2_nit.npy")
A3_nit = np.load("data/A3_nit.npy").reshape((r,r,r))
tensors_nit = (A2_nit,A3_nit)


Phi_pod = np.load("data/Phi_pod.npy")
Psi_pod = Phi_pod.copy()
A2_pod = np.load("data/A2_pod.npy")
A3_pod = np.load("data/A3_pod.npy").reshape((r,r,r))
tensors_pod = (A2_pod,A3_pod)


A2_oi = np.load("data/A2_oi.npy")
A3_oi = np.load("data/A3_oi.npy").reshape((r,r,r))
tensors_oi = (A2_oi,A3_oi)


#%%
time = pool.time
u = np.zeros(r)

plt.figure()
for k in range(n_traj):
    
    mean_en = np.mean(np.linalg.norm(pool.X[k,],axis=0)**2)
    
    # pod
    zpod = Psi_pod.T@pool.X[k,:,0]
    sol = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],zpod,'RK45',t_eval=time,args=(u,) + tensors_pod)).y
    epod = np.linalg.norm(sol - pool.X[k,],axis=0)**2/mean_en
    
    if k == 0:
        plt.plot(time,epod,color=cPOD,linestyle=lPOD,label='POD Gal.')
    else:
        plt.plot(time,epod,color=cPOD,linestyle=lPOD)
    
    # OpInf
    zoi = Psi_pod.T@pool.X[k,:,0]
    sol = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],zoi,'RK45',t_eval=time,args=(u,) + tensors_oi)).y
    eoi = np.linalg.norm(sol - pool.X[k,],axis=0)**2/mean_en
    
    if k == 0:
        plt.plot(time,eoi,color=cOI,linestyle=lOI,label='OpInf')
    else:
        plt.plot(time,eoi,color=cOI,linestyle=lOI)
    
    # NiTROM
    znit = Psi_nit.T@pool.X[k,:,0]
    sol = Phi_nit@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],znit,'RK45',t_eval=time,args=(u,) + tensors_nit)).y
    eoi = np.linalg.norm(sol - pool.X[k,],axis=0)**2/mean_en
    if k == 0:
        plt.plot(time,eoi,color=cOPT,linestyle=lOPT,label='NiTROM')
    else:
        plt.plot(time,eoi,color=cOPT,linestyle=lOPT)
    

ax = plt.gca()
ax.set_yscale('log')
ax.set_ylim([1e-3,ax.get_ylim()[1]])

ax.set_xlabel('Time $t$')
ax.set_ylabel('Error $e(t)$')

plt.legend()
plt.tight_layout()

plt.savefig("./Figures/testing_error.eps",format='eps')
    


#%%

time = dt*np.arange(0,80*n,1)
nsave = 5
eps = 0.1


energies = []

for (k,harmonic) in enumerate([4]):
    
    
    freq = 1.00*harmonic
    tf = np.arange(0,2*np.pi/freq,dt)
    fint = scipy.interpolate.interp1d(tf,eps*np.sin(freq*tf),kind='linear',fill_value='extrapolate')
    
    print("Running simulation with frequency %1.3f"%freq)
    
    
    qic = flow.q_sbf.copy()
    dataf, tsavef = tstep.solver_2D(flow,lops,qic,time,nsave,[0,1,1,0,0,0,0,0],[1],[fint],[2*np.pi/freq],vol_forcing=B)
    energy_true = np.linalg.norm(dataf - flow.q_sbf.reshape(-1,1),axis=0)**2
    

    z0 = np.zeros(r)
    
    # NiTROM
    fnit = np.einsum('i,j',Psi_nit.T@Phi_pre.T@B,eps*np.sin(freq*time))
    fnit = scipy.interpolate.interp1d(time,fnit,kind='linear',fill_value='extrapolate')
    sol_nit = Phi_nit@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=time[::nsave],args=(fnit,) + tensors_nit)).y
    energy_nit = np.linalg.norm(sol_nit,axis=0)**2
    
    # OpInf
    foi = np.einsum('i,j',Psi_pod.T@Phi_pre.T@B,eps*np.sin(freq*time))
    foi = scipy.interpolate.interp1d(time,foi,kind='linear',fill_value='extrapolate')
    sol_oi = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=time[::nsave],args=(foi,) + tensors_oi)).y
    energy_oi = np.linalg.norm(sol_oi,axis=0)**2
    
    # POD
    fpod = np.einsum('i,j',Psi_pod.T@Phi_pre.T@B,eps*np.sin(freq*time))
    fpod = scipy.interpolate.interp1d(time,fpod,kind='linear',fill_value='extrapolate')
    sol_pod = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=time[::nsave],args=(fpod,) + tensors_pod)).y
    energy_pod = np.linalg.norm(sol_pod,axis=0)**2
    
    energy_lst = [energy_true,energy_pod,energy_nit,energy_oi]
    energies.append(energy_lst)


#%%

colors = ['k',cPOD,cOPT,cOI]
lstyle = ['-',lPOD,lOPT,lOI]
fig, ax = plt.subplots(nrows=3,ncols=1)


for k in range (len(energies)):
    
    for (i,vec) in enumerate(energies[k]):
        ax[k].plot(tsavef,vec,color=colors[i],linestyle=lstyle[i])
        if k < len(energies)-1: ax[k].set_xticks([])

ax[-1].set_xlabel('Time $t$')
ax[1].set_ylabel('Energy')

plt.tight_layout()

# plt.savefig("./Figures/freq_response_1.eps",format='eps')

#%%

idx = np.argmin(np.abs(tsavef - 35))

ii = 2

# Ground truth
X, Y, fields = pp.output_fields(flow,dataf[:,idx] - flow.q_sbf)

vmin = np.min(fields[ii]) 
vmax = -vmin

plt.figure()
plt.contourf(X[ii],Y[ii],np.flipud(fields[ii]),levels=100,cmap='bwr',vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.tight_layout()

# plt.savefig("./Figures/snapshot_freqs%1.3f_gt.eps"%(freq),format='eps')


# NiTROM
X, Y, fields = pp.output_fields(flow,Phi_pre@sol_nit[:,idx])

plt.figure()
plt.contourf(X[ii],Y[ii],np.flipud(fields[ii]),levels=100,cmap='bwr',vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.tight_layout()

# plt.savefig("./Figures/snapshot_freqs%1.3f_nitrom.eps"%(freq),format='eps')


# OpInf
X, Y, fields = pp.output_fields(flow,Phi_pre@sol_oi[:,idx])
plt.figure()
plt.contourf(X[ii],Y[ii],np.flipud(fields[ii]),levels=100,cmap='bwr',vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.tight_layout()

# plt.savefig("./Figures/snapshot_freqs%1.3f_opinf.eps"%(freq),format='eps')


# POD
X, Y, fields = pp.output_fields(flow,Phi_pre@sol_pod[:,idx])
plt.figure()
plt.contourf(X[ii],Y[ii],np.flipud(fields[ii]),levels=100,cmap='bwr',vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.tight_layout()

# plt.savefig("./Figures/snapshot_freqs%1.3f_pod.eps"%(freq),format='eps')





