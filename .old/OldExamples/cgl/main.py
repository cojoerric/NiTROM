# import numpy as np
import torch
import matplotlib.pyplot as plt

from NiTROM.Optimization_Functions import classes, nitrom_functions, opinf_functions as opinf_fun, utils
from NiTROM.PyTorch_Functions import gpu_utils, train, integrators
import fom_class_cgl

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.fontsize"] = 14
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2
torch.set_printoptions(precision=8)


cPOD, cOI, cOPT = '#66c2a5', '#fc8d62', '#e78ac3'
lPOD, lOI, lOPT = 'solid', 'dotted', 'dashdot'

device, rank, world_size = gpu_utils.setup_distributed_gpus()
dtype = torch.float64
if rank == 0:
    print(f"Using {world_size} GPU(s) for distributed training.")
    verb = 2
else:
    verb = 0


## Instantiate CGL class and CGL time-stepper class

L = 100
nx = 256
x = torch.linspace(-L/2,L/2,num=nx,device=device,dtype=dtype) 
nu = 1.0*(2 + 0.4*1j)
gamma = 1 - 1j 
mu0 = 0.38
# mu0 = 0.05
mu2 = -0.01
a = 0.1

fom = fom_class_cgl.CGL(x,nu,gamma,mu0,mu2,a)


dt = 1e-2
T = 500
time = dt*torch.arange(0,T//dt,1,device=device,dtype=dtype)
tstep_cgl = fom_class_cgl.time_step_cgl(fom,time)

nsave = 100
tsave = time[::nsave]


traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"


amps = [0.01,1.0,-1.0,0.5]
qIC = fom.B.clone()

q_ = torch.zeros((qIC.shape[0],qIC.shape[-1]*len(amps)), device=device, dtype=dtype)
for k in range (len(amps)):
    q_[:,k*qIC.shape[-1]:(k+1)*qIC.shape[-1]] = amps[k]*qIC

qIC = q_.clone()
n_traj = qIC.shape[-1]

for k in range (n_traj):
        
    print("Running simulation %d/%d"%(k,n_traj))
    
    Qkj, Ykj, tsave = tstep_cgl.time_step(fom,qIC[:,k],nsave)     
    dQkj = torch.zeros_like(Qkj)
    for j in range (Qkj.shape[-1]):
        dQkj[:,j] = fom.evaluate_fom_dynamics(0.0, Qkj[:,j], torch.zeros(Qkj.shape[0],device=device,dtype=dtype))
        
    weight = torch.mean(torch.linalg.norm(Ykj,dim=0)**2)
    
#     np.save(fname_traj%k,Qkj.cpu().numpy())
#     np.save(fname_deriv%k,dQkj.cpu().numpy())
#     np.save(fname_weight%k,[weight.cpu().numpy()])

# np.save(traj_path + "time.npy",tsave.cpu().numpy())

yhat = torch.fft.rfft(fom.compute_output(Qkj),axis=-1)
en = torch.linalg.norm(yhat,dim=0)**2
freqs = torch.arange(0,yhat.shape[-1],1,device=device,dtype=dtype)*2*np.pi/(time[-1]-time[0])

plt.figure()
plt.stem(freqs.cpu(),en.cpu())
plt.gca().set_yscale('log')


## Compute POD ROM

pool_inputs = (n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
               'dtype':dtype,
               'device':device,
               'rank':rank,
               'world_size':world_size
}
pool = classes.pool(*pool_inputs,**pool_kwargs)


n = 2*fom.nx
r = 5               # ROM dimension
poly_comp = [1,3]   # Model with a linear part and a cubic part


Phi_pod = utils.perform_POD(pool,r)
Psi_pod = Phi_pod.clone()
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi_pod,Psi_pod)


plt.figure()
for k in range (pool.n_traj):
    
    print(k)
    y = fom.compute_output(pool.X[k,])
    y = torch.linalg.norm(y,dim=0)**2/pool.weights[k]
    
    plt.plot(pool.time,y)


Phi_bt, Psi_bt = fom_class_cgl.balanced_truncation(fom,tstep_cgl,torch.zeros(n,device=device,dtype=dtype),nsave,r)

# ampsss = [0.01,0.01,0.5,0.5,2.0,2.0,-2.0,-2.0]
plt.figure()
for k in range (n_traj):
    plt.plot(tsave,fom.compute_output(pool.X[k,])[0,])


## Compute Operator Inference model (need NiTROM cost function to select l2 penalty)

which_trajs = torch.arange(0,pool.my_n_traj,1,device=device,dtype=dtype)
which_times = torch.arange(0,pool.n_snapshots,1,device=device,dtype=dtype)
leggauss_deg = 5
nsave_rom = 10

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)



lam = np.logspace(2,9,num=50)
# lam = np.logspace(-2,1,num=20)
cost_oi = []
for (count,l) in enumerate(lam):
    print("Looping over lambda %d/%d"%(count+1,len(lam)))
    tensors_opinf = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,[0.0,l])
    point = (Phi_pod,Psi_pod) + tensors_opinf
    cost_oi.append(cost(*point))
    
# pool.weights = weights

#%%
plt.figure()
plt.plot(lam,cost_oi)

#%%
lambdas = [0.0,lam[np.argmin(cost_oi)]]
tensors_oi = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,lambdas)

print(np.min(cost_oi),lambdas)

#%%
np.save("data/Phi_pod.npy",Phi_pod)
np.save("data/A2_oi.npy",tensors_oi[0])
np.save("data/A4_oi.npy",tensors_oi[1].reshape((r,r**3)))

#%%

line_searcher = myAdaptiveLineSearcher(contraction_factor=0.4,sufficient_decrease=0.1,max_iterations=25,initial_step_size=1)
point = (Phi_pod,Phi_pod) + tensors_oi

k0 = 0
kouter = 20


if k0 == 0:
    costvec_nit = []
    gradvec_nit = []
    
for k in range (k0,k0+kouter):
    
    if np.mod(k,2) == 0:    which_fix = 'fix_bases'
    else:                   which_fix = 'fix_tensors'

    opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
    opt_obj_kwargs = {'which_fix':which_fix}
    opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
    
    print("Optimizing (%d/%d) with which_fix = %s"%(k+1,kouter,opt_obj.which_fix))
    
    cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
    problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
    optimizer = optimizers.ConjugateGradient(max_iterations=5,min_step_size=1e-20,max_time=3600,\
                                              line_searcher=line_searcher,log_verbosity=1,verbosity=2)
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
        
#%%
plt.figure()
plt.plot(costvec_nit)

plt.gca().set_yscale('log')

plt.tight_layout()

    
#%%
opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj_kwargs = {'which_fix':'fix_none'}
opt_obj = classes.optimization_objects(*opt_obj_inputs,**opt_obj_kwargs)
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
check_gradient(problem,x=point)


#%%
Phi_nit = result.point[0]
Psi_nit = result.point[1]
Phi_nit = Phi_nit@scipy.linalg.inv(Psi_nit.T@Phi_nit)
tensors_nit = tuple(result.point[2:])

#%%

Phi_nit = np.load("data/Phi_nit.npy")
Psi_nit = np.load("data/Psi_nit.npy")
A2_nit = np.load("data/A2_nit.npy")
A4_nit = np.load("data/A4_nit.npy").reshape((r,r,r,r))
tensors_nit = (A2_nit,A4_nit)


#%%

Phi_tr = np.load("data/Phi_tr.npy")
Psi_tr = np.load("data/Psi_tr.npy")

tensors_tr, _ = fom.assemble_petrov_galerkin_tensors(Phi_tr,Psi_tr)

#%%

vec = np.random.randn(2)
F = qIC[:,:2]@vec 
F = 0.1*F/np.linalg.norm(F) 
F = F.reshape(-1,1)


wf = 0.648/2
wf = 0.04*2
Tf = 2*np.pi/wf
tf = dt*np.arange(0,Tf//dt,1)

fu = scipy.interpolate.interp1d(tf,np.outer(F,np.sin(wf*tf)),kind='linear',fill_value='extrapolate')
_, sol_fom, tsave = tstep_cgl.time_step(fom,0*qIC[:,0],nsave,fu,Tf)  

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
plt.show()