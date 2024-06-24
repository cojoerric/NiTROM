#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 19:57:05 2024

@author: alberto
"""

import matplotlib.pyplot as plt
import numpy as np 
import classes_cavity


from mpi4py import MPI

plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],
                     "font.size":12})

import sys
import pymanopt.manifolds as manifolds

sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

import nitrom_functions 
import classes
import opinf_functions as opinf_fun

#%% Define FOM-related stuff

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

#%% Load training data

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"
amps = np.load(traj_path + "amps_train.npy")


Phi_pre = np.load(traj_path + "Phi_pre.npy")
n_traj = len(amps)
n = Phi_pre.shape[-1]
r = 50                  # ROM dimension
poly_comp = [1,2]       # Model with a linear part and a cubic part


pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)


#%%
which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 10

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrrr = manifolds.Euclidean(r,r,r,r)

M = manifolds.Product([Gr,St,Euc_rr,Euc_rrrr])
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)


Phi_pod = np.zeros((n,r))
Phi_pod[:r,:r] = np.eye(r)
Psi_pod = Phi_pod.copy()


weights = pool.weights.copy()
pool.weights *= pool.n_traj*pool.n_snapshots

lam = np.logspace(-4,-1,num=30)
cost_oi = []
for (count,l) in enumerate(lam):
    tensors_opinf = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,[0.0,l])
    point = (Phi_pod,Psi_pod) + tensors_opinf
    cost_oi.append(cost(*point))
    print("Computing OpInf with lambda = %1.2e (%d/%d). Cost = %1.7e"%(l,count + 1,len(lam),cost_oi[-1]))
    
pool.weights = weights

#%%
lambdas = [0.0,lam[np.argmin(cost_oi)]]

weights = pool.weights.copy()
pool.weights *= pool.n_traj*pool.n_snapshots

tensors_oi = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,lambdas)
point = (Phi_pod,Psi_pod) + tensors_oi
print(cost(*point))

print(lambdas,np.min(cost_oi))

pool.weights = weights

#%%

np.save("data/Phi_pod.npy",Phi_pod)
np.save("data/A2_oi.npy",tensors_oi[0])
np.save("data/A3_oi.npy",tensors_oi[1].reshape((r,r**2)))


#%%

Phi = Phi_pre@Phi_pod 
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi, Phi, B, [0,0,1,0,0,0,0,0])
np.save("data/A2_pod.npy",tensors_pod[0])
np.save("data/A3_pod.npy",tensors_pod[1].reshape((r,r**2)))




