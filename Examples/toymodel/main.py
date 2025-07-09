import numpy as np 
import scipy
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers

from NiTROM_GPU.Optimization_Functions import classes, nitrom_functions
from NiTROM_GPU.PyManopt_Functions.my_pymanopt_classes import myAdaptiveLineSearcher
from NiTROM_GPU.PyTorch_Functions import gpu_utils
import fom_class_pytorch

plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')
torch.set_printoptions(precision=8)

cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'

device, rank, world_size = gpu_utils.setup_distributed_gpus()
if rank == 0:
    print(f"Using {world_size} GPU(s) for distributed training.")
    verb = 2
else:
    verb = 0

n = 3
n_traj = 4
beta = 20.0
diag_vec = torch.tensor([-1.0,-2.0,-5.0],device=device,dtype=torch.float64)
A2 = torch.diag(diag_vec)
A3 = torch.zeros((3,3,3), device=device)
diag_vec2 = torch.tensor([beta,beta,0.0],device=device,dtype=torch.float64)
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
pool_kwargs = {'fname_steady_forcing':fname_forcing,
               'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
               'device':device,
               'rank':rank,
               'world_size':world_size
}
pool = classes.pool(*pool_inputs,**pool_kwargs)

r = 2               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part

#%% Compute NiTROM model 

which_trajs = torch.arange(0,pool.n_traj,1,device=device)
which_times = torch.arange(0,pool.n_snapshots,1,device=device)
leggauss_deg = 5
nsave_rom = 2

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrr = manifolds.Euclidean(r,r,r)

M = manifolds.Product([Gr,St,Euc_rr,Euc_rrr])
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)

line_searcher = myAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.85,max_iterations=25,initial_step_size=1)
optimizer = optimizers.ConjugateGradient(max_iterations=40,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=1,verbosity=verb)

point = [None]*4
if rank == 0:
    point[0] = np.load(traj_path + "Phi_pod.npy")
    point[1] = np.load(traj_path + "Psi_pod.npy")
    point[2] = np.load(traj_path + "A2.npy")
    point[3] = np.load(traj_path + "A3.npy")
dist.broadcast_object_list(point,src=0)
point = tuple(point)

result = optimizer.run(problem,initial_point=point)

if rank == 0:
    Phi_nit = result.point[0]
    Psi_nit = result.point[1]
    Phi_nit = Phi_nit@scipy.linalg.inv(Psi_nit.T@Phi_nit)
    tensors_nit = tuple(result.point[2:])

    np.save('results/Phi_nit_gpu_fast.npy',Phi_nit)
    np.save('results/Psi_nit_gpu_fast.npy',Psi_nit)
    np.save('results/A2_nit_gpu_fast.npy',tensors_nit[0])
    np.save('results/A3_nit_gpu_fast.npy',tensors_nit[1])

    itervec_nit = result.log["iterations"]["iteration"]
    costvec_nit = result.log["iterations"]["cost"]
    gradvec_nit = result.log["iterations"]["gradient_norm"]

    np.save('results/nitrom_gpu_itervec_fast.npy',itervec_nit)
    np.save('results/nitrom_gpu_costvec_fast.npy',costvec_nit)
    np.save('results/nitrom_gpu_gradvec_fast.npy',gradvec_nit)

    plt.figure()
    plt.plot(itervec_nit,costvec_nit,color=cOPT,linestyle=lOPT,label='NiTROM')

    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xlabel('Conj. gradient iteration')
    ax.set_ylabel('Cost')

    plt.legend()
    plt.tight_layout()
    plt.show()

gpu_utils.cleanup_distributed()