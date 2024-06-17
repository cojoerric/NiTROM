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
plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')

sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

from my_pymanopt_classes import myAdaptiveLineSearcher
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
nu = 0.0*(2 + 0.4*1j)
gamma = 1 - 1j 
# mu0 = 0.38
mu0 = 0.05
mu2 = -0.01
a = 0.1

fom = fom_class_cgl.CGL(x,nu,gamma,mu0,mu2,a)


dt = 1e-2
T = 500
time = dt*np.arange(0,T//dt,1)
tstep_cgl = fom_class_cgl.time_step_cgl(fom,time)

nsave = 100
tsave = time[::nsave]

# plt.figure()
# plt.plot(fom.x,fom.B[:fom.nx,0])
# plt.plot(fom.x,fom.C[0,:fom.nx])


#%%
traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"


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
    
    np.save(fname_traj%k,Qkj)
    np.save(fname_deriv%k,dQkj)
    np.save(fname_weight%k,[weight])

np.save(traj_path + "time.npy",tsave)

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

#%%
plt.figure()
for k in range (n_traj):
    plt.plot(tsave,fom.compute_output(pool.X[k,])[0,])


#%% Compute Operator Inference model (need NiTROM cost function to select l2 penalty)

which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 10

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
# opt_obj_kwargs = {'stab_promoting_pen':1e-2,'stab_promoting_tf':20,'stab_promoting_ic':(np.random.randn(r),)}


opt_obj = classes.optimization_objects(*opt_obj_inputs)


St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrrr = manifolds.Euclidean(r,r,r,r)

M = manifolds.Product([Gr,St,Euc_rr,Euc_rrrr])
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)




#%%
# lam = np.logspace(2,9,num=20)
lam = np.logspace(-5,-1,num=20)
cost_oi = []
for (count,l) in enumerate(lam):
    print("Looping over lambda %d/%d"%(count+1,len(lam)))
    tensors_opinf = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,[0.0,l])
    point = (Phi_pod,Psi_pod) + tensors_opinf
    cost_oi.append(cost(*point))

#%%
plt.figure()
plt.plot(lam,cost_oi)

#%%
lambdas = [0.0,lam[np.argmin(cost_oi)]]
tensors_oi = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,lambdas)


np.save("data/Phi_pod.npy",Phi_pod)
np.save("data/A2_oi.npy",tensors_oi[0])
np.save("data/A4_oi.npy",tensors_oi[1].reshape((r,r**3)))

#%%

# line_searcher = myAdaptiveLineSearcher(contraction_factor=0.4,sufficient_decrease=0.1,max_iterations=25,initial_step_size=1)
# point = (Phi_pod,Phi_pod) + tensors_oi

# k0 = 0
# kouter = 20


# if k0 == 0:
#     costvec_nit = []
#     gradvec_nit = []
    
# for k in range (k0,k0+kouter):
    
#     if np.mod(k,2) == 0:    which_fix = 'fix_bases'
#     else:                   which_fix = 'fix_tensors'

#     opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
#     opt_obj_kwargs = {'which_fix':which_fix}
#     opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
    
#     print("Optimizing (%d/%d) with which_fix = %s"%(k+1,kouter,opt_obj.which_fix))
    
#     cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
#     problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
#     optimizer = optimizers.ConjugateGradient(max_iterations=5,min_step_size=1e-20,max_time=3600,\
#                                              line_searcher=line_searcher,log_verbosity=1,verbosity=2)
#     result = optimizer.run(problem,initial_point=point)
#     point = result.point
    
#     itervec_nit_k = result.log["iterations"]["iteration"]
#     costvec_nit_k = result.log["iterations"]["cost"]
#     gradvec_nit_k = result.log["iterations"]["gradient_norm"]
    
#     if k == 0:    
#         costvec_nit.extend(costvec_nit_k) 
#         gradvec_nit.extend(gradvec_nit_k) 
#     else:         
#         costvec_nit.extend(costvec_nit_k[1:]) 
#         gradvec_nit.extend(gradvec_nit_k[1:]) 
        
# #%%
# plt.figure()
# plt.plot(costvec_nit)

# plt.gca().set_yscale('log')

# plt.tight_layout()

    
# #%%
# opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
# opt_obj_kwargs = {'which_fix':'fix_none'}
# opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
# cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
# problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
# check_gradient(problem,x=point)


# #%%
# Phi_nit = result.point[0]
# Psi_nit = result.point[1]
# Phi_nit = Phi_nit@scipy.linalg.inv(Psi_nit.T@Phi_nit)
# tensors_nit = tuple(result.point[2:])

# #%%

# vec = np.random.randn(2)
# F = qIC[:,:2]@vec 
# F = 0.05*F/np.linalg.norm(F) 
# F = F.reshape(-1,1)


# wf = 0.648
# Tf = 2*np.pi/wf
# tf = dt*np.arange(0,Tf//dt,1)

# fu = scipy.interpolate.interp1d(tf,np.outer(F,np.sin(wf*tf)),kind='linear',fill_value='extrapolate')
# _, sol_fom, tsave = tstep_cgl.time_step(fom,0*qIC[:,0],nsave,fu,Tf)  

# #%%
# time = tstep_cgl.time
# fu = scipy.interpolate.interp1d(time,np.outer(Psi_pod.T@F,np.sin(wf*time)),kind='linear',fill_value='extrapolate')
# sol_pod = fom.compute_output(Phi_pod)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=tsave,args=(fu,) + tensors_pod)).y

# fu = scipy.interpolate.interp1d(time,np.outer(Psi_pod.T@F,np.sin(wf*time)),kind='linear',fill_value='extrapolate')
# sol_oi = fom.compute_output(Phi_pod)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=tsave,args=(fu,) + tensors_oi)).y

# fu = scipy.interpolate.interp1d(time,np.outer(Psi_nit.T@F,np.sin(wf*time)),kind='linear',fill_value='extrapolate')
# sol_nit = fom.compute_output(Phi_nit)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=tsave,args=(fu,) + tensors_nit)).y

# plt.figure()
# plt.plot(tsave,sol_fom[0,],color='k',linewidth=2)
# plt.plot(tsave,sol_pod[0,],color=cPOD,linestyle=lPOD,linewidth=2)
# plt.plot(tsave,sol_oi[0,],color=cOI,linestyle=lOI,linewidth=2)
# plt.plot(tsave,sol_nit[0,],color=cOPT,linestyle=lOPT,linewidth=2)


    
