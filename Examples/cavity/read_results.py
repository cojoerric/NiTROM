import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI
import time_steppers as tstep

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys, os

# sys.path.append("/Users/alberto/Documents/SIAM_nitrom/OptimizationFunctions/")
import post_process as pp

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from pymanopt.tools.diagnostics import check_gradient


plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')

sys.path.append(os.path.abspath("../../OptimizationFunctions/"))


from my_pymanopt_classes import myAdaptiveLineSearcher

import classes
import nitrom_functions 
import opinf_functions as opinf_fun
import troop_functions
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

flow = classes_cavity.flow_parameters(Lx,Ly,Nx,Ny,Re)

n = 400
dt = 1.0/n

lops = classes_cavity.linear_operators_2D(flow,dt)
flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))
fom = classes_cavity.fom_class(flow,lops)
fom.assemble_forcing_profile(0.95, 0.05)

#%%

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"


amps = np.load(traj_path + "amps.npy")
Phi_pre = np.load(traj_path + "Phi_pre.npy")
n_traj = len(amps)

n = Phi_pre.shape[-1]

#%%
pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)


r = 50               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part


#%%
Phi_pod = np.zeros((n,r))
Phi_pod[:r,:r] = np.eye(r)
Psi_pod = Phi_pod.copy()


tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi_pre@Phi_pod,Phi_pre@Psi_pod)

#%%
which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 15

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,[1,2])
opt_obj_kwargs = {'stab_promoting_pen':0.0,'stab_promoting_tf':150,'stab_promoting_ic':(np.random.randn(r),)}


opt_obj = classes.optimization_objects(*opt_obj_inputs)

St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrr = manifolds.Euclidean(r,r,r)

M = manifolds.Product([Gr,St,Euc_rr,Euc_rrr])
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)


#%%

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

print(pool.weights)

weights = pool.weights.copy()
pool.weights *= pool.n_traj*pool.n_snapshots

print(pool.weights)
lam = np.logspace(3,4,num=20)
cost_oi = []
for (count,l) in enumerate(lam):
    tensors_opinf = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,[0.0,l])
    point = (Phi_pod,Psi_pod) + tensors_opinf
    cost_oi.append(cost(*point))
    print("Computing OpInf with lambda = %1.2e (%d/%d). Cost = %1.7e"%(l,count + 1,len(lam),cost_oi[-1]))
    
pool.weights = weights

print(pool.weights)

#%%
# plt.figure()
# plt.plot(lam[0:],cost_oi[0:])

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)


weights = pool.weights.copy()
pool.weights *= pool.n_traj*pool.n_snapshots


tensors_oi = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,[0,0.01])
point = (Phi_pod,Psi_pod) + tensors_pod
print(cost(*point))

pool.weights = weights
# np.save("data/A2_oi.npy",tensors_oi[0])
# np.save("data/A3_oi.npy",tensors_oi[1].reshape((r,r**2)))

#%%

Phi_nit = np.load("data/Phi_nit.npy")
Psi_nit = np.load("data/Psi_nit.npy")
A2_nit = np.load("data/A2_nit.npy")
A3_nit = np.load("data/A3_nit.npy").reshape((r,r,r))

tensors_nit = (A2_nit,A3_nit)

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
    plt.plot(time,epod,color=cPOD,linestyle=lPOD)
    
    # OpInf
    zoi = Psi_pod.T@pool.X[k,:,0]
    sol = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],zoi,'RK45',t_eval=time,args=(u,) + tensors_oi)).y
    eoi = np.linalg.norm(sol - pool.X[k,],axis=0)**2/mean_en
    plt.plot(time,eoi,color=cOI,linestyle=lOI)
    
    # NiTROM
    znit = Psi_nit.T@pool.X[k,:,0]
    sol = Phi_nit@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],znit,'RK45',t_eval=time,args=(u,) + tensors_nit)).y
    eoi = np.linalg.norm(sol - pool.X[k,],axis=0)**2/mean_en
    plt.plot(time,eoi,color=cOPT,linestyle=lOPT)
    
    
ax = plt.gca()
ax.set_yscale('log')
ax.set_ylim([1e-3,ax.get_ylim()[1]])

    
#%%
# np.save("data/A2_oi.npy",tensors_oi[0])
# np.save("data/A3_oi.npy",tensors_oi[1].reshape((r,r**2)))

#%%

flow.ff = fom.f.copy()

nn = 400
dt = 1.0/nn
freq = 5
tf = np.arange(0,2*np.pi*freq,0.001)
time = dt*np.arange(0,40*nn,1)
nsave = 50

eps = 0.05
fint = scipy.interpolate.interp1d(tf,eps*np.sin(freq*tf),kind='linear',fill_value='extrapolate')


# qic = np.random.randn(len(flow.q_sbf))
# qic /= np.linalg.norm(qic)
# qic = 0.5*flow.ff

dataf, tsavef = tstep.nonlinear_solver_2D(flow,lops,flow.q_sbf,time,nsave,fint,2*np.pi*freq)

energy_true = np.linalg.norm(dataf - flow.q_sbf.reshape(-1,1),axis=0)**2


#%%

z0 = np.zeros(r)


# NiTROM
fnit = np.einsum('i,j',Psi_nit.T@Phi_pre.T@flow.ff,eps*np.sin(freq*time))
fnit = scipy.interpolate.interp1d(time,fnit,kind='linear',fill_value='extrapolate')
sol_nit = Phi_nit@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=time[::nsave],args=(fnit,) + tensors_nit)).y
energy_nit = np.linalg.norm(sol_nit,axis=0)**2


# OpInf
foi = np.einsum('i,j',Psi_pod.T@Phi_pre.T@flow.ff,eps*np.sin(freq*time))
foi = scipy.interpolate.interp1d(time,foi,kind='linear',fill_value='extrapolate')
sol_oi = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=time[::nsave],args=(foi,) + tensors_oi)).y
energy_oi = np.linalg.norm(sol_oi,axis=0)**2


# POD
fpod = np.einsum('i,j',Psi_pod.T@Phi_pre.T@flow.ff,eps*np.sin(freq*time))
fpod = scipy.interpolate.interp1d(time,fpod,kind='linear',fill_value='extrapolate')
sol_pod = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=time[::nsave],args=(fpod,) + tensors_pod)).y
energy_pod = np.linalg.norm(sol_pod,axis=0)**2


plt.figure()
plt.plot(tsavef,energy_true,color='k')
# plt.plot(tsavef,energy_pod,color=cPOD,linestyle=lPOD)
# plt.plot(tsavef,energy_oi,color=cOI,linestyle=lOI)
plt.plot(tsavef,energy_nit,color=cOPT,linestyle=lOPT)

#%%

idx = np.argmin(np.abs(tsavef - 20))

ii = 2
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


X, Y, fields = pp.output_fields(flow,Phi_pre@sol_nit[:,idx])

plt.figure()
plt.contourf(X[ii],Y[ii],np.flipud(fields[ii]),levels=100,cmap='bwr',vmin=vmin,vmax=vmax)

ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.tight_layout()

X, Y, fields = pp.output_fields(flow,Phi_pre@sol_oi[:,idx])

plt.figure()
plt.contourf(X[ii],Y[ii],np.flipud(fields[ii]),levels=100,cmap='bwr',vmin=vmin,vmax=vmax)

ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.tight_layout()

X, Y, fields = pp.output_fields(flow,Phi_pre@sol_pod[:,idx])

plt.figure()
plt.contourf(X[ii],Y[ii],np.flipud(fields[ii]),levels=100,cmap='bwr',vmin=vmin,vmax=vmax)

ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.tight_layout()

#%%

qic = np.random.randn(len(flow.q_sbf))
qic /= np.linalg.norm(qic)
dataf, tsavef = tstep.nonlinear_solver_2D(flow,lops,flow.q_sbf + qic,time,nsave)

#%%
dhat = np.fft.rfft(dataf - flow.q_sbf.reshape(-1,1),axis=-1)/len(tsavef)
freqvec = 2*np.pi/(tsavef[-1])*np.arange(0,dhat.shape[-1],1)

en = np.linalg.norm(dhat,axis=0)**2

#%%
plt.figure()
plt.stem(freqvec,en)
plt.gca().set_yscale('log')


