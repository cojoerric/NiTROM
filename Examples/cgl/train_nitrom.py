import numpy as np 
import scipy 
from mpi4py import MPI

import sys, os

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers

sys.path.append(os.path.abspath("../../OptimizationFunctions/"))

from my_pymanopt_classes import myAdaptiveLineSearcher

import classes
import nitrom_functions 
import fom_class_cgl


# Define FOM
L = 100
nx = 256
x = np.linspace(-L/2,L/2,num=nx,endpoint=True) 
nu = 2 + 0.4*1j 
gamma = 1 - 1j 
mu0 = 0.38
mu2 = -0.01
a = 0.1

fom = fom_class_cgl.CGL(x,nu,gamma,mu0,mu2,a)


dt = 1e-2
T = 1000
time = dt*np.arange(0,T//dt,1)
tstep_cgl = fom_class_cgl.time_step_cgl(fom,time)

nsave = 100
tsave = time[::nsave]



traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.txt"
fname_weight = traj_path + "weight_%03d.txt"
fname_deriv = traj_path + "deriv_%03d.txt"
fname_time = traj_path + "time.txt"


n_traj = 8

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)


n = 2*fom.nx
r = 5               # ROM dimension
poly_comp = [1,3]   # Model with a linear part and a quadratic part


# Phi_pod = np.load("data/Phi_pod.npy")
# A2_oi = np.load("data/A2_oi.npy")
# A4_oi = np.load("data/A4_oi.npy").reshape((r,r,r,r))
# tensors_oi = (A2_oi,A4_oi)

Phi_nit = np.load("data/Phi_nit.npy")
Psi_nit = np.load("data/Psi_nit.npy")
A2_nit = np.load("data/A2_nit.npy")
A4_nit = np.load("data/A4_nit.npy").reshape((r,r,r,r))
tensors_nit = (A2_nit,A4_nit)


which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 10

St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrrr = manifolds.Euclidean(r,r,r,r)
M = manifolds.Product([Gr,St,Euc_rr,Euc_rrrr])

line_searcher = myAdaptiveLineSearcher(contraction_factor=0.4,sufficient_decrease=0.1,max_iterations=25,initial_step_size=1)
# point = (Phi_pod,Phi_pod) + tensors_oi
point = (Phi_nit,Psi_nit) + tensors_nit


if pool.rank == 0:  verb = 2
else:               verb = 0

k0 = 0
kouter = 50


if k0 == 0:
    costvec_nit = []
    gradvec_nit = []
    
for k in range (k0,k0+kouter):
    
    if np.mod(k,2) == 0:    which_fix = 'fix_bases'
    else:                   which_fix = 'fix_tensors'

    opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
    opt_obj_kwargs = {'which_fix':which_fix}
    opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
    
    if pool.rank == 0:
        print("Optimizing (%d/%d) with which_fix = %s"%(k+1,kouter,opt_obj.which_fix))
    
    cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
    problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
    optimizer = optimizers.ConjugateGradient(max_iterations=5,min_step_size=1e-20,max_time=3600,\
                                             line_searcher=line_searcher,log_verbosity=1,verbosity=verb)
    result = optimizer.run(problem,initial_point=point)
    point = result.point
    
    itervec_nit_k = result.log["iterations"]["iteration"]
    costvec_nit_k = result.log["iterations"]["cost"]
    gradvec_nit_k = result.log["iterations"]["gradient_norm"]
    
    if k == 0:    
        costvec_nit.extend(costvec_nit_k) 
        gradvec_nit.extend(gradvec_nit_k) 
    else:         
        costvec_nit.extend(costvec_nit_k[1:]) 
        gradvec_nit.extend(gradvec_nit_k[1:]) 

# check_gradient(problem,x=[Phi_pod,Phi_pod,*tensors_oi])
# check_gradient(problem,x=result.point)

Phi_nit = result.point[0]
Psi_nit = result.point[1]
Phi_nit = Phi_nit@scipy.linalg.inv(Psi_nit.T@Phi_nit)
tensors_nit = tuple(result.point[2:])

if pool.rank == 0:
    np.save("data/Phi_nit.npy",Phi_nit)
    np.save("data/Psi_nit.npy",Psi_nit)
    np.save("data/A2_nit.npy",tensors_nit[0])
    np.save("data/A4_nit.npy",tensors_nit[1].reshape((r,r**3)))









