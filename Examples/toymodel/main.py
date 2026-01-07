import torch
import matplotlib.pyplot as plt

from NiTROM.Optimization_Functions import classes, nitrom_models as model
from NiTROM.PyTorch_Functions import gpu_utils, train, integrators
import fom_class_pytorch

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

cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'

device, rank, world_size = gpu_utils.setup_distributed_gpus()
dtype = torch.float64
if rank == 0:
    print(f"Using {world_size} devices for distributed training.")
    print(f"Device: {device}")

n = 3
n_traj = 4
beta = 20.0
diag_vec = torch.tensor([-1.0,-2.0,-5.0],device=device,dtype=dtype)
A2 = torch.diag(diag_vec)
A3 = torch.zeros((3,3,3), device=device,dtype=dtype)
diag_vec2 = torch.tensor([beta,beta,0.0],device=device,dtype=dtype)
A3[:,:,-1] = torch.diag(diag_vec2)
B = torch.ones((3,1),device=device,dtype=dtype)
C = torch.ones((1,3),device=device,dtype=dtype)
fom = fom_class_pytorch.full_order_model(A2,A3,B,C,device=device,dtype=dtype)

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
               'dtype':dtype,
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

N = pool.n_snapshots*pool.n_traj
X = torch.zeros((pool.X.shape[1],N), device=device, dtype=dtype)
for i in range (pool.n_traj):
    X[:,i*pool.n_snapshots:(i+1)*pool.n_snapshots] = pool.X[i,]
phi_pod, _, _ = torch.linalg.svd(X,full_matrices=False)
phi_pod = phi_pod[:,:r]
psi_pod = phi_pod.clone()
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(phi_pod,psi_pod)
A_pod, H_pod = tensors_pod

init = {"Phi":phi_pod,
        "Psi":psi_pod,
        "A2":A_pod,
        "A3":H_pod
}

params = model.NitromParams(pool, r, poly_comp, init=init, requires_grad=True)
model = model.NitromModel(params, opt_obj, fom).to(device)
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=10, line_search_fn='strong_wolfe')

# do_gradcheck = True
# if do_gradcheck and rank == 0:
#     utils.finite_difference_gradcheck(model, n_samples=4, eps=1e-5, seed=0)

num_epochs = 20
model, history = train.train_model(
    model,
    pool,
    optimizer,
    num_epochs,
    log_every=1,
    manifold_retraction="qr",
)

phi_nit = model.params.Phi.detach()
psi_nit = model.params.Psi.detach()
A_nit = model.params.A2.detach()
H_nit = model.params.A3.detach()
tensors_nit = (A_nit, H_nit)


# Plot errors
max_val = 5/20
betas = torch.rand(100, dtype=dtype, device=device)*0.999*max_val
t_eval = torch.linspace(0, 10, steps=100, device=device, dtype=dtype)
error_pod = torch.zeros_like(t_eval)
error_nit = torch.zeros_like(t_eval)

for k in range(len(betas)):
    u = betas[k]*torch.ones(n, device=device, dtype=dtype)
    x0 = torch.zeros(n, device=device, dtype=dtype)
    z0 = torch.zeros(r, device=device, dtype=dtype)

    sol = integrators.my_rk4_adaptive(fom.evaluate_fom_dynamics, t_eval, x0, args=(u,))
    id_ss = torch.tensor([-betas[k]/(-1 + 4*betas[k]), -betas[k]/(-2 + 4*betas[k]), betas[k]/5], device=device, dtype=dtype)
    weight = torch.norm(fom.compute_output(id_ss))**2

    sol_pod_r = integrators.my_rk4_adaptive(opt_obj.evaluate_rom_rhs, t_eval, z0, args=(psi_pod.T@u,) + tensors_pod)
    sol_pod = phi_pod @ sol_pod_r
    error_pod += torch.norm(C @ (sol_pod - sol), dim=0)**2 / weight / len(betas)

    sol_nit_r = integrators.my_rk4_adaptive(opt_obj.evaluate_rom_rhs, t_eval, z0, args=(psi_nit.T@u,) + tensors_nit)
    sol_nit = phi_nit @ sol_nit_r
    error_nit += torch.norm(C @ (sol_nit - sol), dim=0)**2 / weight / len(betas)

plt.figure()
plt.semilogy(t_eval.cpu(), error_pod.cpu(), label='POD', color=cPOD, linestyle=lPOD)
plt.semilogy(t_eval.cpu(), error_nit.cpu(), label='NiTROM', color=cOPT, linestyle=lOPT)
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.savefig('figures/error_20')