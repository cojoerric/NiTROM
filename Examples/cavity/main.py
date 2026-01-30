import torch
import numpy as np
import matplotlib.pyplot as plt

from NiTROM.Optimization_Functions import classes, nitrom_models as nit_model, opinf_models as oi_model, opinf_closed_form as oi_cf, utils
from NiTROM.PyTorch_Functions import gpu_utils, train, integrators
import classes_cavity

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


device, rank, world_size = gpu_utils.setup_distributed_gpus()
dtype = torch.float64

# Enable faster fp32 matmuls on Ampere+ (no effect on CPU)
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

if rank == 0:
    print(f"Using {world_size} GPU(s) for distributed training.")
    print(f"Device: {device}")

use_ic = 1
run_pod = 0
run_opinf = 0
run_opinf_gs = 0
run_nitrom = 0
run_nitrom_gs = 1

Lx = 1
Ly = 1
Nx = 100
Ny = 100

dx = Lx/Nx
dy = Ly/Ny
Re = 8300

flow = classes_cavity.flow_class(Lx,Ly,Nx,Ny,Re)
integrator = integrators.my_rk4_adaptive

n = 400
dt = 1.0/n

lops = classes_cavity.linear_operators_2D(flow,dt)
flow.q_sbf = np.load("bflow_Re%d_Nx%d_Ny%d.npy"%(Re,Nx,Ny))
fom = classes_cavity.fom_class(flow,lops)

fom.assemble_forcing_profile(0.95,0.05)
B = fom.f.copy()

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

amps = np.load(traj_path + "amps.npy")
phi_pre = np.load(traj_path + "phi_pre.npy")
n = phi_pre.shape[-1]

n_traj = len(amps)
pool_inputs = (n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
               'dtype':dtype,
               'device':device,
               'rank':rank,
               'world_size':world_size
}
pool = classes.pool(*pool_inputs,**pool_kwargs)

r = 50               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part

which_trajs = torch.arange(0,pool.n_traj,1,device=device)
which_times = torch.arange(0,pool.n_snapshots,1,device=device)
leggauss_deg = 5
nsave_rom = 2

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

phi_pod = np.eye(n, r)
psi_pod = phi_pod.copy()
phi_tot = phi_pre @ phi_pod
psi_tot = phi_pre @ psi_pod
phi_pod = torch.tensor(phi_pod, device=device, dtype=dtype)
psi_pod = torch.tensor(psi_pod, device=device, dtype=dtype)


## Load previous run to use as IC
if use_ic:
    print("\nLoading IC...")
    A_ic = torch.tensor(np.load('results/A_pod.npy'), device=device, dtype=dtype)
    H_ic = torch.tensor(np.load('results/H_pod.npy'), device=device, dtype=dtype)


## Compute POD model
if run_pod:
    print("\nComputing POD Model...")
    tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(phi_tot, psi_tot, B, [0,0,1,0,0,0,0,0])
    np.save('results/A_pod.npy', tensors_pod[0])
    np.save('results/H_pod.npy', tensors_pod[1])
    tensors_pod = tuple([torch.tensor(tensor, device=device, dtype=dtype) for tensor in tensors_pod])
    A_pod, H_pod = tensors_pod

    if not use_ic:
        A_ic = A_pod.clone()
        H_ic = H_pod.clone()


## Compute OpInf model
if run_opinf:
    print("\nComputing OpInf Model...")
    weights = pool.weights.clone()
    pool.weights *= pool.n_traj*pool.n_snapshots

    lam = np.logspace(-4,-1,num=30)
    cost_oi = []
    for (count,l) in enumerate(lam):
        tensors_opinf = oi_cf.operator_inference(pool, phi_pod, poly_comp, [0.0,l])
        init = {
            "Phi": phi_pod,
            "Psi": psi_pod,
            "A2": tensors_opinf[0],
            "A3": tensors_opinf[1],
        }
        params_oi = nit_model.NitromParams(pool, r, poly_comp, init=init, requires_grad=True).to(device)
        model_oi = nit_model.NitromModel(params_oi, opt_obj, fom, integrator).to(device)
        cost_oi.append(model_oi().item())
        print(f"  Lambda {count+1}/{len(lam)}: {l:.4e}, Cost: {cost_oi[-1]:.6e}")
        
    pool.weights = weights

    lambdas = [0.0,lam[np.argmin(cost_oi)]]
    print(lambdas, np.min(cost_oi))

    weights = pool.weights.clone()
    pool.weights *= pool.n_traj*pool.n_snapshots

    tensors_oi = oi_cf.operator_inference(pool, phi_pod, poly_comp, lambdas)
    A_oi, H_oi = tensors_oi

    pool.weights = weights

    np.save('results/A_oi.npy', A_oi.cpu().numpy())
    np.save('results/H_oi.npy', H_oi.cpu().numpy())


## Compute globally stable OpInf model
if run_opinf_gs:
    print("\nTraining OpInf (GS) Model...")
    init = utils.create_intitial_guess(A_ic, H_ic, r=r)
    params_oi = oi_model.OpinfParams_GloballyStable(pool, r, poly_comp, init=init, requires_grad=True).to(device)
    model_oi = oi_model.OpinfModel(phi_pod, params_oi, opt_obj).to(device)
    optimizer_oi = torch.optim.LBFGS(model_oi.parameters(), lr=1.0, max_iter=20, history_size=10, line_search_fn='strong_wolfe')

    model_oi, history = train.train_model(
        model_oi,
        pool,
        optimizer_oi,
        num_epochs=2500,
        log_every=100,
    )

    Qhat = model_oi.params.Qhat.detach()
    Jhat = model_oi.params.Jhat.detach()
    Rhat = model_oi.params.Rhat.detach()
    Hhat = model_oi.params.Hhat.detach()
    A_oi_gs, H_oi_gs = utils.construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]

    np.save('results/A_oi_gs.npy', A_oi_gs.cpu().numpy())
    np.save('results/H_oi_gs.npy', H_oi_gs.cpu().numpy())


## Compute NiTROM model
if run_nitrom:
    print("\nTraining NiTROM Model...")
    init = {
        "Phi": phi_pod,
        "Psi": psi_pod,
        "A2": A_ic,
        "A3": H_ic,
    }
    for factor in np.arange(1, 17):
        print(f"\n  Training with factor {factor}...")
        which_times = torch.arange(0, 10*factor, 1, device=device)
        opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
        opt_obj = classes.optimization_objects(*opt_obj_inputs)

        params_nit = nit_model.NitromParams(pool, r, poly_comp, init=init, requires_grad=True).to(device)
        model_nit = nit_model.NitromModel(params_nit, opt_obj, fom, integrator).to(device)
        optimizer_nit = torch.optim.AdamW(model_nit.parameters(), lr=1e-2, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nit, mode='min', factor=0.5, patience=20, threshold=1e-4, min_lr=1e-6)

        model_nit, history = train.train_model(
            model_nit,
            pool,
            optimizer_nit,
            num_epochs=200,
            log_every=10,
            manifold_retraction="qr",
            scheduler=scheduler,
            manifold_lr=1e-3,
        )

        phi_nit = model_nit.params.Phi.detach()
        psi_nit = model_nit.params.Psi.detach()
        phi_nit = phi_nit @ torch.linalg.inv(psi_nit.T @ phi_nit)
        A_nit = model_nit.params.A2.detach()
        H_nit = model_nit.params.A3.detach()

        init = {
            "Phi": phi_nit,
            "Psi": psi_nit,
            "A2": A_nit,
            "A3": H_nit,
        }

        np.save('results/phi_nit.npy', phi_nit.cpu().numpy())
        np.save('results/psi_nit.npy', psi_nit.cpu().numpy())
        np.save('results/A_nit.npy', A_nit.cpu().numpy())
        np.save('results/H_nit.npy', H_nit.cpu().numpy())


## Compute globally stable NiTROM model
if run_nitrom_gs:
    print("\nTraining NiTROM (GS) Model...")
    if run_nitrom:
        init = utils.create_intitial_guess(A_nit, H_nit, r=r)
        init["Phi"] = phi_nit.clone()
        init["Psi"] = psi_nit.clone()
    else:
        init = utils.create_intitial_guess(A_ic, H_ic, r=r)
        init["Phi"] = phi_pod.clone()
        init["Psi"] = psi_pod.clone()

    for factor in np.arange(1, 17):
        print(f"\n  Training with factor {factor}...")
        which_times = torch.arange(0, 10*factor, 1, device=device)
        opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
        opt_obj = classes.optimization_objects(*opt_obj_inputs)

        params_nit_gs = nit_model.NitromParams_GloballyStable(pool, r, poly_comp, init=init, requires_grad=True).to(device)
        model_nit_gs = nit_model.NitromModel(params_nit_gs, opt_obj, fom, integrator).to(device)
        optimizer_nit_gs = torch.optim.AdamW(model_nit_gs.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nit_gs, mode='min', factor=0.5, patience=20, threshold=1e-4, min_lr=1e-6)
        # optimizer_nit_gs = torch.optim.LBFGS(model_nit_gs.parameters(), lr=1.0, max_iter=20, history_size=10)

        model_nit_gs, history = train.train_model(
            model_nit_gs,
            pool,
            optimizer_nit_gs,
            num_epochs=200,
            log_every=10,
            manifold_retraction="qr",
            scheduler=scheduler,
            manifold_lr=1e-3,
        )

        phi_nit_gs = model_nit_gs.params.Phi.detach()
        psi_nit_gs = model_nit_gs.params.Psi.detach()
        phi_nit_gs = phi_nit_gs @ torch.linalg.inv(psi_nit_gs.T @ phi_nit_gs)
        Qhat = model_nit_gs.params.Qhat.detach()
        Jhat = model_nit_gs.params.Jhat.detach()
        Rhat = model_nit_gs.params.Rhat.detach()
        Hhat = model_nit_gs.params.Hhat.detach()
        A_nit_gs, H_nit_gs = utils.construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]

        init = {
            "Phi": phi_nit_gs,
            "Psi": psi_nit_gs,
            "Qhat": Qhat,
            "Jhat": Jhat,
            "Rhat": Rhat,
            "Hhat": Hhat,
        }
        np.save('results/phi_nit_gs.npy', phi_nit_gs.cpu().numpy())
        np.save('results/psi_nit_gs.npy', psi_nit_gs.cpu().numpy())
        np.save('results/A_nit_gs.npy', A_nit_gs.cpu().numpy())
        np.save('results/H_nit_gs.npy', H_nit_gs.cpu().numpy())