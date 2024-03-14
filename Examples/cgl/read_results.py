import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from pymanopt.tools.diagnostics import check_gradient
from pymanopt.optimizers.line_search import myAdaptiveLineSearcher

plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')

sys.path.append("/Users/alberto/Documents/SIAM_nitrom/OptimizationFunctions/")
import classes
import nitrom_functions 
import opinf_functions as opinf_fun
import troop_functions
import fom_class_cgl


cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'


#%% Instantiate CGL class and CGL time-stepper class

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
fname_forcing = traj_path + "forcing_%03d.txt"
fname_deriv = traj_path + "deriv_%03d.txt"
fname_time = traj_path + "time.txt"

n_traj = 8
n = 2*fom.nx
r = 5               # ROM dimension
poly_comp = [1,3]   # Model with a linear part and a quadratic part

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)


which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 10

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
# opt_obj_kwargs = {'stab_promoting_pen':1e-2,'stab_promoting_tf':20,'stab_promoting_ic':(np.random.randn(r),)}

opt_obj = classes.optimization_objects(*opt_obj_inputs)


#%%

r = 5

Phi_nit = np.load("data/Phi_nit.npy")
Psi_nit = np.load("data/Psi_nit.npy")
A2_nit = np.load("data/A2_nit.npy")
A4_nit = np.load("data/A4_nit.npy").reshape((r,r,r,r))
tensors_nit = (A2_nit,A4_nit)


Phi_pod = np.load("data/Phi_pod.npy")
Psi_pod = Phi_pod.copy()
A2_oi = np.load("data/A2_oi.npy")
A4_oi = np.load("data/A4_oi.npy").reshape((r,r,r,r))
tensors_oi = (A2_oi,A4_oi)


tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi_pod,Phi_pod)

#%%
vec = np.random.randn(2)
F = fom.B@vec 
F = 0.05*F/np.linalg.norm(F) 
F = F.reshape(-1,1)


wf = 2*0.648
Tf = 2*np.pi/wf
tf = dt*np.arange(0,Tf//dt,1)

fu = scipy.interpolate.interp1d(tf,np.outer(F,np.sin(wf*tf)),kind='linear',fill_value='extrapolate')
_, sol_fom, tsave = tstep_cgl.time_step(fom,np.zeros(n),nsave,fu,Tf)  

#%%
time = tstep_cgl.time
fu = scipy.interpolate.interp1d(time,np.outer(Psi_pod.T@F,np.sin(wf*time)),kind='linear',fill_value='extrapolate')
sol_pod = fom.compute_output(Phi_pod)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=tsave,args=(fu,) + tensors_pod)).y

fu = scipy.interpolate.interp1d(time,np.outer(Psi_pod.T@F,np.sin(wf*time)),kind='linear',fill_value='extrapolate')
sol_oi = fom.compute_output(Phi_pod)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=tsave,args=(fu,) + tensors_oi)).y

fu = scipy.interpolate.interp1d(time,np.outer(Psi_nit.T@F,np.sin(wf*time)),kind='linear',fill_value='extrapolate')
sol_nit = fom.compute_output(Phi_nit)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=tsave,args=(fu,) + tensors_nit)).y

plt.figure()
plt.plot(tsave,sol_fom[0,],color='k',linewidth=2)
plt.plot(tsave,sol_pod[0,],color=cPOD,linestyle=lPOD,linewidth=2)
plt.plot(tsave,sol_oi[0,],color=cOI,linestyle=lOI,linewidth=2)
plt.plot(tsave,sol_nit[0,],color=cOPT,linestyle=lOPT,linewidth=2)


    


















