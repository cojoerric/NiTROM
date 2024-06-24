import matplotlib.pyplot as plt
import numpy as np 
import time_steppers as tstep
import post_process as pp
import classes_cavity
import time as tlib

import scipy.linalg as sciplin
from mpi4py import MPI

plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],
                     "font.size":12})

import sys
import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers

sys.path.append("../../PyManopt_Functions/")
sys.path.append("../../Optimization_Functions/")

from my_pymanopt_classes import myAdaptiveLineSearcher
import nitrom_functions 
import classes
import opinf_functions as opinf_fun

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


fom = classes_cavity.fom_class(flow,lops)
fom.assemble_forcing_profile(0.95,0.05)
B = fom.f.copy()



#%%

# ff = fom.evaluate_fom_fullrhs_wallpert(0*flow.q_sbf)
X, Y, fields = pp.output_fields(flow,B)

color_map = plt.cm.get_cmap('bwr')

idx = 1
vmin = np.min(fields[idx]) 
vmax = -vmin

plt.figure()
plt.contourf(X[idx],Y[idx],np.flipud(fields[idx]),levels=100,cmap=color_map,vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar()
# ax.set_xticks([0,0.25,0.5,0.75,1.0])
# ax.set_yticks([0,0.25,0.5,0.75,1.0])

# plt.savefig("./Figures/bflow_Re%d.eps"%Re,format='eps')
# plt.savefig("./Figures/bflow_Re%d.png"%Re)

u = fields[0]
v = fields[1]

print(np.max(fields[idx]),np.min(fields[idx]))


#%%
nsave = 100
time = dt*np.arange(0,n*40,1)
tsave = time[::nsave]

# amps = np.asarray([-1e-1,-5e-2,-1e-2,-1e-3,1e-3,1e-2,5e-2,1e-1])
amps = [-1.0, -0.25, -0.05, 0.01, 0.05, 0.25, 1.0]
# amps = np.random.uniform(-1,1,25)


bc_coefs = [0,0,1,0,0,0,0,0]
Q = np.zeros((flow.szu + flow.szv,len(amps)*len(tsave)))
energy = np.zeros((len(amps),len(tsave)))

for k in range (len(amps)):
    
    t0 = tlib.time()
    print("Generating trajectory %d/%d"%(k+1,len(amps)))
    qic = flow.q_sbf + amps[k]*B
    data, _ = tstep.solver_2D(flow,lops,qic,time,nsave,bc_coefs)
    data -= flow.q_sbf.reshape(-1,1)
    
    Q[:,k*len(tsave):(k+1)*len(tsave)] = data 
    energy[k,] = np.linalg.norm(data,axis=0)**2
    t1 = tlib.time() - t0
    print("Execution time = %1.3f [min]"%(t1/60))
    

#%%
plt.figure()
for k in range (len(amps)):
    plt.plot(tsave,energy[k,],'k')
    
# plt.gca().set_ylim(0,50)

#%%

U, S, _ = sciplin.svd(Q,full_matrices=False)
Slo = 1 - np.cumsum(S**2)/np.sum(S**2)

#%%
plt.figure()
plt.plot(Slo,'o')

ax = plt.gca()
ax.set_yscale('log')

#%%

# Phi_pre = U[:,:200]
Phi_pre = np.load("./trajectories/Phi_pre.npy")

#%%
traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"


#%%
for k in range (len(amps)):
    
    print("Saving trajectory %d/%d"%(k+1,len(amps)))
    
    data = Q[:,k*len(tsave):(k+1)*len(tsave)]
    ddata = np.zeros_like(data)
    for j in range (data.shape[-1]):
        ddata[:,j] = fom.evaluate_fom_dynamics(data[:,j],bc_coefs,[0,0,0,0,0,0,0,0])
        
    data = Phi_pre.T@data
    ddata = Phi_pre.T@ddata
    weight = np.mean(np.linalg.norm(data,axis=0)**2)
    
    np.save(fname_traj%k,data)
    np.save(fname_deriv%k,ddata)
    np.save(fname_weight%k,[weight])
    
np.save(fname_time,tsave)
np.save(traj_path + "amps.npy",amps)
np.save(traj_path + "Phi_pre.npy",Phi_pre)


#%% Compute Operator Inference model (need NiTROM cost function to select l2 penalty)


pool_inputs = (MPI.COMM_WORLD, len(amps), fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

# Phi_pre = np.load(traj_path + "Phi_pre.npy")
n = Phi_pre.shape[-1]
r = 50                  # ROM dimension
poly_comp = [1,2]       # Model with a linear part and a cubic part


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
Phi_pod = np.zeros((n,r))
Phi_pod[:r,:r] = np.eye(r)
Psi_pod = Phi_pod.copy()
# lam = np.logspace(5,9,num=20)
# lam = np.logspace(-5,1,num=20)
# cost_oi = []
# for (count,l) in enumerate(lam):
#     print("Looping over lambda %d/%d"%(count+1,len(lam)))
#     tensors_opinf = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,[0.0,l])
#     point = (Phi_pod,Psi_pod) + tensors_opinf
#     cost_oi.append(cost(*point))
    
print(pool.weights)

weights = pool.weights.copy()
pool.weights *= pool.n_traj*pool.n_snapshots

print(pool.weights)
lam = np.logspace(-5,1,num=20)
cost_oi = []
for (count,l) in enumerate(lam):
    tensors_opinf = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,[0.0,l])
    point = (Phi_pod,Psi_pod) + tensors_opinf
    cost_oi.append(cost(*point))
    print("Computing OpInf with lambda = %1.2e (%d/%d). Cost = %1.7e"%(l,count + 1,len(lam),cost_oi[-1]))
    
pool.weights = weights

print(pool.weights)

#%%
plt.figure()
plt.plot(lam,cost_oi)

#%%
lambdas = [0.0,lam[np.argmin(cost_oi)]]
tensors_oi = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,lambdas)

print(lambdas,np.min(cost_oi))

#%%
np.save("data/Phi_pod.npy",Phi_pod)
np.save("data/A2_oi.npy",tensors_oi[0])
np.save("data/A3_oi.npy",tensors_oi[1].reshape((r,r**2)))


#%%
X, Y, fields = pp.output_fields(flow,U[:,1])

color_map = plt.cm.get_cmap('bwr')

idx = 2
vmin = np.min(fields[idx]) 
vmax = -vmin

plt.figure()
plt.contourf(X[idx],Y[idx],np.flipud(fields[idx]),levels=100,cmap=color_map,vmin=vmin,vmax=vmax)
ax = plt.gca()
ax.set_aspect('equal')
plt.colorbar()
# ax.set_xticks([0,0.25,0.5,0.75,1.0])
# ax.set_yticks([0,0.25,0.5,0.75,1.0])

# plt.savefig("./Figures/bflow_Re%d.eps"%Re,format='eps')
# plt.savefig("./Figures/bflow_Re%d.png"%Re)

#%%
r = 50
# Phi = U[:,:r]
Phi = Phi_pre[:,:r]

#%%
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi, Phi, B, bc_coefs)
#%%
Phi_pod = np.zeros((n,r))
Phi_pod[:r,:r] = np.eye(r)
Psi_pod = Phi_pod.copy()
point = (Phi_pod,Phi_pod) + tensors_pod
print(cost(*point))

#%%
np.save("data/Phi_pod.npy",Phi_pod)
np.save("data/A2_pod.npy",tensors_pod[0])
np.save("data/A3_pod.npy",tensors_pod[1].reshape((r,r**2)))
#%%

vec = np.random.randn(flow.szu + flow.szv)
vec /= np.linalg.norm(vec)
vec = Phi@(Phi.T@vec)

# Method 1
rhs = Phi@(Phi.T@fom.evaluate_fom_dynamics(vec))

# Method 2
z = Phi.T@vec 
fz = Phi@(np.einsum('ij,j',tensors_pod[0],z) + np.einsum('ijk,j,k',tensors_pod[1],z,z))

diff = fz - rhs
print(np.linalg.norm(diff))

# Method 3

fq = Phi@(Phi.T@(fom.evaluate_fom_fullrhs(vec) - fom.evaluate_fom_fullrhs(0*vec)))

diff = fq - rhs
print(np.linalg.norm(diff))

#%%

vec = np.zeros(flow.rowsu)

y0, yf = 0.8, 0.95
l = yf - y0
for i in range (len(vec)):
    if flow.y[i] >= y0 and flow.y[i] <= yf:
        vec[i] = np.sin(2*np.pi/l*(flow.y[i] - y0))
        
plt.figure()
plt.plot(flow.y,vec,'o')

#%%
import numba_operators as numba_ops
qout = numba_ops.divergence_boundary_conditions_2D_znmf(flow.Re,flow.x,flow.y,vec)

qout = qout.reshape((flow.rowsp,flow.colsp))

plt.figure()
plt.plot(flow.y,-vec/(flow.x[1]-flow.x[0]),'o')
plt.plot(flow.y,qout[:,-1],'x')
    
