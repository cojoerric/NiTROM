import numpy as np 
import scipy 
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from pymanopt.tools.diagnostics import check_gradient
from pymanopt.optimizers.line_search import myAdaptiveLineSearcher


sys.path.append("/scratch1/09103/padovan3/NiTROM_tests/OptimizationFunctions")
import classes
import nitrom_functions 
import opinf_functions as opinf_fun
import troop_functions
import fom_class_cgl


cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'


#%% # Instantiate CGL class and CGL time-stepper class

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
T = 500
time = dt*np.arange(0,T//dt,1)
tstep_cgl = fom_class_cgl.time_step_cgl(fom,time)

nsave = 100
tsave = time[::nsave]


#%%
traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.txt"
fname_weight = traj_path + "weight_%03d.txt"
fname_deriv = traj_path + "deriv_%03d.txt"
fname_time = traj_path + "time.txt"


amps = [0.01,0.1,1.0,-1.0]
qIC = fom.B.copy()

q_ = np.zeros((qIC.shape[0],qIC.shape[-1]*len(amps)))
for k in range (len(amps)):
    q_[:,k*qIC.shape[-1]:(k+1)*qIC.shape[-1]] = amps[k]*qIC

qIC = q_.copy()
n_traj = qIC.shape[-1]

for k in range (n_traj):
        
    print("Running simulation %d/%d"%(k,n_traj))
    
    Qkj, Ykj, tsave = tstep_cgl.time_step(fom,qIC[:,k],nsave)     
    dQkj = np.zeros_like(Qkj)
    for j in range (Qkj.shape[-1]):
        dQkj[:,j] = fom.evaluate_fom_dynamics(0.0, Qkj[:,j], np.zeros(Qkj.shape[0]))
        
    weight = np.mean(np.linalg.norm(Ykj,axis=0)**2)
    
    np.savetxt(fname_traj%k,Qkj)
    np.savetxt(fname_deriv%k,dQkj)
    np.savetxt(fname_weight%k,[weight])

np.savetxt(traj_path + "time.txt",tsave)
#%% Compute POD ROM

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)


n = 2*fom.nx
r = 5               # ROM dimension
poly_comp = [1,3]   # Model with a linear part and a quadratic part


Phi_pod, sig = opinf_fun.perform_POD(pool,r)
Psi_pod = Phi_pod.copy()
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi_pod,Psi_pod)


#%% Compute Operator Inference model (need NiTROM cost function to select l2 penalty)

lambdas = [0.0,1e9]
tensors_oi = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,lambdas)


np.save("data/Phi_pod.npy",Phi_pod)
np.save("data/A2_oi.npy",tensors_oi[0])
np.save("data/A4_oi.npy",tensors_oi[1].reshape((r,r**3)))


    
