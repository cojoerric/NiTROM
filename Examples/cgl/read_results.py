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


#%% Instantiate CGL class and CGL time-stepper class

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


if mu0 == 0.38:     case = 1
else:               case = 2

#%%

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

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


Phi_tr = np.load("data/Phi_tr.npy")
Psi_tr = np.load("data/Psi_tr.npy")

tensors_tr, _ = fom.assemble_petrov_galerkin_tensors(Phi_tr,Psi_tr)


tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi_pod,Phi_pod)

#%%

itervec_tr = np.load("data/itervec_tr.npy")
costvec_tr = np.load("data/costvec_tr.npy")

itervec_nit = np.load("data/itervec_nit.npy")
costvec_nit = np.load("data/costvec_nit.npy")


plt.figure()
plt.plot(itervec_tr,costvec_tr,color=cTR,linestyle=lTR,label='TrOOP')
plt.plot(itervec_nit,costvec_nit,color=cOPT,linestyle=lOPT,label='NiTROM')

ax = plt.gca()
ax.set_yscale('log')
ax.set_xlabel('Conj. gradient iteration')
ax.set_ylabel('Cost')

ax.set_box_aspect(0.25)

plt.legend()
plt.tight_layout()

plt.savefig("Figures/convergence_plot_case%d.eps"%case,format='eps')

#%%
vec = np.random.randn(2)
# vec = [1,0]
F = fom.B@vec 
F = 0.05*F/np.linalg.norm(F) 
F = F.reshape(-1,1)


wf = 0.648*2
# wf = 0.044
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

fu = scipy.interpolate.interp1d(time,np.outer(Psi_tr.T@F,np.sin(wf*time)),kind='linear',fill_value='extrapolate')
sol_tr = fom.compute_output(Phi_tr)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=tsave,args=(fu,) + tensors_tr)).y

# plt.figure()
# plt.plot(tsave,sol_fom[0,],color='k',linewidth=2)
# plt.plot(tsave,sol_pod[0,],color=cPOD,linestyle=lPOD,linewidth=2)
# plt.plot(tsave,sol_oi[0,],color=cOI,linestyle=lOI,linewidth=2)
# plt.plot(tsave,sol_nit[0,],color=cOPT,linestyle=lOPT,linewidth=2)


fig, ax = plt.subplots(nrows=2,ncols=1)

ax[0].plot(tsave,sol_fom[0,],'k')
ax[0].plot(tsave,sol_tr[0,],color=cTR,linestyle=lTR,linewidth=2)
ax[0].plot(tsave,sol_oi[0,],color=cOI,linestyle=lOI,linewidth=2)


ax[1].plot(tsave,sol_fom[0,],'k')
ax[1].plot(tsave,sol_pod[0,],color=cPOD,linestyle=lPOD,linewidth=2)
ax[1].plot(tsave,sol_nit[0,],color=cOPT,linestyle=lOPT,linewidth=2)


ax[0].set_ylabel(r'$Re\left(y(t)\right)$')
ax[0].set_xticks([])
ax[0].set_xlim([-0.2,200])
ax[1].set_ylabel(r'$Re\left(y(t)\right)$')
ax[1].set_xlabel('Time $t$')
ax[1].set_xlim([-0.2,200])
ax[1].set_ylim(ax[0].get_ylim())


plt.tight_layout()


plt.savefig('./Figures/cgl_sin_response_case%d_freq%1.3f.eps'%(case,wf),format='eps')


#%%
amps = np.random.uniform(-1,1,25)
qIC_test = fom.B.copy()

q_ = np.zeros((qIC_test.shape[0],qIC_test.shape[-1]*len(amps)))
for k in range (len(amps)):
    q_[:,k*qIC_test.shape[-1]:(k+1)*qIC_test.shape[-1]] = amps[k]*qIC_test

qIC_test = q_.copy()
ntraj = qIC_test.shape[-1]


Q_test = np.zeros((ntraj,n,len(tsave)))
en_test = np.zeros((ntraj,len(tsave)))

weights = np.zeros(ntraj)
for k in range (ntraj):
        
    print("Running simulation %d/%d"%(k,ntraj))
    
    Qkj, Ykj, tsave = tstep_cgl.time_step(fom,qIC_test[:,k],nsave)     
    Q_test[k,:,:] = Qkj
    en_test[k,]  = np.linalg.norm(Ykj,axis=0)**2

#%%

error_pod = np.zeros((ntraj,len(tsave)))
error_oi = np.zeros((ntraj,len(tsave)))
error_opt = np.zeros((ntraj,len(tsave)))
error_tr = np.zeros((ntraj,len(tsave)))

u = np.zeros(r)

for traj in range (ntraj):

    ytrue = fom.compute_output(Q_test[traj,:,:])
    energy_ytrue = np.mean(np.linalg.norm(ytrue,axis=0)**2)
    
    
    z0 = Phi_pod.T@Q_test[traj,:,0]
    ypod = fom.compute_output(Phi_pod)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=tsave,args=(u,) + tensors_pod)).y
    error_pod[traj,] = np.linalg.norm(ytrue - ypod,axis=0)**2/energy_ytrue
    
    z0 = Phi_pod.T@Q_test[traj,:,0]
    yoi = fom.compute_output(Phi_pod)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=tsave,args=(u,) + tensors_oi)).y
    error_oi[traj,] = np.linalg.norm(ytrue - yoi,axis=0)**2/energy_ytrue
    
    
    z0 = Psi_nit.T@Q_test[traj,:,0]
    yopt = fom.compute_output(Phi_nit)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=tsave,args=(u,) + tensors_nit)).y
    error_opt[traj,] = np.linalg.norm(ytrue - yopt,axis=0)**2/energy_ytrue
    # print(np.linalg.norm(error_opt[traj,]),traj)
    
    
    z0 = Psi_tr.T@Q_test[traj,:,0]
    ytr = fom.compute_output(Phi_tr)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=tsave,args=(u,) + tensors_tr)).y
    error_tr[traj,] = np.linalg.norm(ytrue - ytr,axis=0)**2/energy_ytrue


#%%
plt.figure()

plt.plot(tsave[:],np.mean(error_pod,axis=0),linestyle=lPOD,color=cPOD,label='POD Gal.')
plt.plot(tsave[:],np.mean(error_oi,axis=0),linestyle=lOI,color=cOI,label='OpInf')
plt.plot(tsave[:],np.mean(error_tr,axis=0),linestyle=lTR,color=cTR,label='TrOOP')
plt.plot(tsave[:],np.mean(error_opt,axis=0),linestyle=lOPT,color=cOPT,label='NiTROM')
    
ax = plt.gca()
ax.set_xlabel('Time $t$')
ax.set_ylabel('Average error $e(t)$')
ax.set_yscale('log')
# ax.set_ylim(1e-5,ax.get_ylim()[1])
# ax.grid('on')

plt.legend(loc='lower left')
plt.tight_layout()

# plt.savefig('./Figures/cgl_avg_error_plot_case%d.eps'%case,format='eps')

#%%

traj = 15

ytrue = fom.compute_output(Q_test[traj,:,:])


z0 = Phi_pod.T@Q_test[traj,:,0]
ypod = fom.compute_output(Phi_pod)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=tsave,args=(u,) + tensors_pod)).y

z0 = Phi_pod.T@Q_test[traj,:,0]
yoi = fom.compute_output(Phi_pod)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=tsave,args=(u,) + tensors_oi)).y


z0 = Psi_nit.T@Q_test[traj,:,0]
yopt = fom.compute_output(Phi_nit)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=tsave,args=(u,) + tensors_nit)).y


z0 = Psi_tr.T@Q_test[traj,:,0]
ytr = fom.compute_output(Phi_tr)@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],z0,'RK45',t_eval=tsave,args=(u,) + tensors_tr)).y


fig, ax = plt.subplots(nrows=2,ncols=1)

ax[0].plot(tsave,ytrue[0,],'k')
ax[0].plot(tsave,ytr[0,],color=cTR,linestyle=lTR,linewidth=2)
ax[0].plot(tsave,yoi[0,],color=cOI,linestyle=lOI,linewidth=2)


ax[1].plot(tsave,ytrue[0,],'k')
ax[1].plot(tsave,ypod[0,],color=cPOD,linestyle=lPOD,linewidth=2)
ax[1].plot(tsave,yopt[0,],color=cOPT,linestyle=lOPT,linewidth=2)


ax[0].set_ylabel(r'$Re\left(y(t)\right)$')
ax[0].set_xticks([])
ax[0].set_xlim([-0.2,200])
ax[1].set_ylabel(r'$Re\left(y(t)\right)$')
ax[1].set_xlabel('Time $t$')
ax[1].set_xlim([-0.2,200])
ax[1].set_ylim(ax[0].get_ylim())


plt.tight_layout()


# plt.savefig('./Figures/cgl_impulse_response_case%d.eps'%case,format='eps')


#%%

yhat = np.fft.rfft(ytrue,axis=-1)
en = np.linalg.norm(yhat,axis=0)**2
freqs = np.arange(0,yhat.shape[-1],1)*2*np.pi/(time[-1]-time[0])

plt.figure()
plt.stem(freqs,en)
plt.gca().set_yscale('log')

#%%
import scipy as sp
evals, evecs = sp.linalg.eig(fom.A.toarray())


plt.figure()
plt.plot(evals.real,evals.imag,'o')















