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
import nitrom_functions 
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


# Compute the effect of the BC on the flow (then reset the flow class)
flow.create_side_wall_forcing_profile(0.45,0.55,'ur')
fom = classes_cavity.fom_class(flow,lops)
B = fom.evaluate_full_fom_dynamics(np.zeros(flow.szu + flow.szv),[0,1e-4,1,0,0,0,0,0])/1e-4
flow.ur *= 0
fom = classes_cavity.fom_class(flow,lops)


fom.assemble_forcing_profile(0.95,0.05)
B = fom.f.copy()

#%%

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"


amps = np.load(traj_path + "amps_train.npy")
Phi_pre = np.load(traj_path + "Phi_pre.npy")
n_traj = len(amps)

n = Phi_pre.shape[-1]

#%%
pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)


r = 50                # ROM dimension
poly_comp = [1,2]       # Model with a linear part and a quadratic part


Phi_pod = np.zeros((n,r))
Phi_pod[:r,:r] = np.eye(r)
Psi_pod = Phi_pod.copy()

# A2_oi = np.load("data/A2_oi.npy")
# A3_oi = np.load("data/A3_oi.npy").reshape((r,r,r))
# tensors_oi = (A2_oi,A3_oi)
# point = (Phi_pod,Phi_pod) + tensors_oi

Phi_nit = np.load("data/Phi_nit.npy")
Psi_nit = np.load("data/Psi_nit.npy")
A2_nit = np.load("data/A2_nit.npy")
A3_nit = np.load("data/A3_nit.npy").reshape((r,r,r))
tensors_nit = (A2_nit,A3_nit)
point = (Phi_nit,Psi_nit) + tensors_nit


#%%

rand_vec = np.random.randn(r)
line_searcher = myAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.1,max_iterations=25,initial_step_size=1)

for factor in np.arange(16,17,1):
    
    which_trajs = np.arange(0,pool.my_n_traj,1)
    which_times = np.arange(0,10*factor,1)
    leggauss_deg = 5
    nsave_rom = 15
    if pool.rank == 0:
        print("---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("---------- Running with factor %d -----------------------"%factor)
    
    opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,[1,2])
    opt_obj_kwargs = {'stab_promoting_pen':1e-3,'stab_promoting_tf':100,'stab_promoting_ic':(rand_vec,)}
    
    opt_obj = classes.optimization_objects(*opt_obj_inputs)#,**opt_obj_kwargs)
    
    St = manifolds.Stiefel(n,r)
    Gr = manifolds.Grassmann(n,r)
    Euc_rr = manifolds.Euclidean(r,r)
    Euc_rrr = manifolds.Euclidean(r,r,r)
    M = manifolds.Product([Gr,St,Euc_rr,Euc_rrr])
    
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
        # which_fix = 'fix_bases'
    
        opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
        opt_obj_kwargs = {'which_fix':which_fix,'stab_promoting_pen':1e-3,'stab_promoting_tf':100,'stab_promoting_ic':(rand_vec,)}
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
        
        if pool.rank == 0:
            evals__, _ = scipy.linalg.eig(point[2])
            idces_ev = np.argwhere(evals__.real >= 0).reshape(-1)
            print(evals__[idces_ev])
            
    Phi_nit = result.point[0]
    Psi_nit = result.point[1]
    Phi_nit = Phi_nit@scipy.linalg.inv(Psi_nit.T@Phi_nit)
    tensors_nit = tuple(result.point[2:])
    
    
    
    np.save("data/Phi_nit.npy",Phi_nit)
    np.save("data/Psi_nit.npy",Psi_nit)
    np.save("data/A2_nit.npy",tensors_nit[0])
    np.save("data/A3_nit.npy",tensors_nit[1].reshape((r,r**2)))


# check_gradient(problem,x=[Phi_pod,Phi_pod,*tensors_oi])
# check_gradient(problem,x=result.point)






















