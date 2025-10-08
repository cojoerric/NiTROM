#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:31:02 2024

@author: alberto
"""

import sys, os

def set_cpu_threads(n):
  n = str(int(n))
  for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
  ):
    os.environ[k] = n
 
# If you check 'htop -u <user>', you should see the nb. of threads used equal to 'nt'
nt = 1
set_cpu_threads(nt)


import numpy as np 
import scipy 
from mpi4py import MPI



import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers

sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

from my_pymanopt_classes import myAdaptiveLineSearcher

import classes
import troop_functions 
import fom_class_cgl


# Define FOM
L = 100
nx = 256
x = np.linspace(-L/2,L/2,num=nx,endpoint=True) 
nu = 1.0*(2 + 0.4*1j)
gamma = 1 - 1j 
mu0 = 0.38
# mu0 = 0.05
mu2 = -0.01
a = 0.1

fom = fom_class_cgl.CGL(x,nu,gamma,mu0,mu2,a)


dt = 1e-2
T = 500
time = dt*np.arange(0,T//dt,1)
tstep_cgl = fom_class_cgl.time_step_cgl(fom,time)

nsave = 100
tsave = time[::nsave]



traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"


n_traj = 8

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)


n = 2*fom.nx
r = 5               # ROM dimension
poly_comp = [1,3]   # Model with a linear part and a quadratic part


Phi_pod = np.load("data/Phi_bt.npy")
# Psi_pod = Phi_pod.copy()
Psi_pod = np.load("data/Psi_bt.npy")


if pool.rank == 0:  verb = 2
else:               verb = 0


which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 10

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)


line_searcher = myAdaptiveLineSearcher(contraction_factor=0.4,sufficient_decrease=0.1,max_iterations=25,initial_step_size=1)
optimizer = optimizers.ConjugateGradient(max_iterations=200,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=1,verbosity=verb)
point = (Phi_pod,Phi_pod)

Gr = manifolds.Grassmann(n,r)
M = manifolds.Product([Gr,Gr])
cost_troop, grad_troop, _ = troop_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
problem = pymanopt.Problem(M,cost_troop,euclidean_gradient=grad_troop)


result = optimizer.run(problem,initial_point=(Phi_pod,Psi_pod))

Phi_tr = result.point[0]
Psi_tr = result.point[1]
Phi_tr = Phi_tr@scipy.linalg.inv(Psi_tr.T@Phi_tr)

tensors_tr, _ = fom.assemble_petrov_galerkin_tensors(Phi_tr,Psi_tr)

itervec_tr = result.log["iterations"]["iteration"]
costvec_tr = result.log["iterations"]["cost"]
gradvec_tr = result.log["iterations"]["gradient_norm"]


np.save("data/Phi_tr.npy",Phi_tr)
np.save("data/Psi_tr.npy",Psi_tr)

np.save("data/costvec_tr.npy",costvec_tr)
np.save("data/itervec_tr.npy",itervec_tr)
