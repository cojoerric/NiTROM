import numpy as np 
import torch
import matplotlib.pyplot as plt
import time

import pymanopt.manifolds as manifolds

import cProfile, pstats
pr = cProfile.Profile()

plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')

from NiTROM_GPU.Optimization_Functions import classes, nitrom_functions
import fom_class_pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=8)


n = 3
n_traj = 4
beta = 20.0
diag_vec = torch.tensor([-1.0,-2.0,-5.0],device=device)
A2 = torch.diag(diag_vec)
A3 = torch.zeros((3,3,3), device=device)
diag_vec2 = torch.tensor([beta,beta,0.0],device=device)
A3[:,:,-1] = torch.diag(diag_vec2)
B = torch.ones((3,1),device=device,dtype=torch.float64)
C = torch.ones((1,3),device=device,dtype=torch.float64)
fom = fom_class_pytorch.full_order_model(A2,A3,B,C)

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

pool_inputs = (n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_steady_forcing':fname_forcing,'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.pool(*pool_inputs,**pool_kwargs)

r = 2               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part


#%% Compute NiTROM model 

which_trajs = torch.arange(0,pool.n_traj,1)
which_times = torch.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 3

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,[1,2])

opt_obj = classes.optimization_objects(*opt_obj_inputs)

St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrr = manifolds.Euclidean(r,r,r)

M = manifolds.Product([Gr,St,Euc_rr,Euc_rrr])
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)

weights = pool.weights.detach().clone()
pool.weights *= pool.n_traj*pool.n_snapshots

Phi_pod = np.load(traj_path + "Phi_pod.npy")
Psi_pod = np.load(traj_path + "Psi_pod.npy")
A2 = np.load(traj_path + "A2.npy")
A3 = np.load(traj_path + "A3.npy")
tensors_pod = (A2,A3)
point = (Phi_pod,Psi_pod) + tensors_pod

t1 = time.time()
pr.enable()
grad_val = grad(*point)
pr.disable()
t2 = time.time()

pool.weights = weights

print(f"Gradient evaluation time: {t2 - t1:.4f} seconds")
norm = 0
for tensor in grad_val:
    norm += np.linalg.norm(tensor)
print("Gradient norm:", norm)

pr.dump_stats("gradient_profiling.prof")
stats = pstats.Stats(pr).sort_stats('cumtime')
stats.print_stats(20)   # top 20 slowest functions