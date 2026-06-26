import numpy as np
import scipy
import matplotlib.pyplot as plt
import time as tlib
from mpi4py import MPI

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers

from NiTROM.Optimization_Functions import classes, nitrom_functions, opinf_functions as opinf_fun, opinf_functions_grad as opinf_fun_grad
from NiTROM.Optimization_Functions.utils import create_initial_guess, construct_operators
from NiTROM.PyManopt_Functions.my_pymanopt_classes import myAdaptiveLineSearcher
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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

if rank == 0:
    print(f"Using {world_size} MPI process(es) for CPU training.")
    verb = 2
else:
    verb = 0

use_ic = 0
run_pod = 1
run_opinf = 1
run_opinf_gs = 1
run_nitrom = 1
run_nitrom_gs = 1

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

traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

amps = np.load(traj_path + "amps.npy")
phi_pre = np.load(traj_path + "phi_pre.npy")
n = phi_pre.shape[-1]

n_traj = len(amps)
pool_inputs = (comm, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_weights':fname_weight,
               'fname_derivs':fname_deriv,
}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

r = 30               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part

which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 15

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
opt_obj = classes.optimization_objects(*opt_obj_inputs)

phi_pod = np.eye(n, r)
psi_pod = phi_pod.copy()
phi_tot = phi_pre @ phi_pod
psi_tot = phi_pre @ psi_pod


## Load previous run to use as IC
if use_ic:
    if rank == 0:
        print("\nLoading IC...")
    phi_ic = np.load('results/phi_nit.npy')
    psi_ic = np.load('results/psi_nit.npy')
    A_ic = np.load('results/A_nit.npy')
    H_ic = np.load('results/H_nit.npy')
    Qhat_ic = np.load('results/Qhat_oi_gs.npy')
    Jhat_ic = np.load('results/Jhat_oi_gs.npy')
    Rhat_ic = np.load('results/Rhat_oi_gs.npy')
    Hhat_ic = np.load('results/Hhat_oi_gs.npy')


## Compute POD model
if run_pod:
    if rank == 0:
        print("\nComputing POD Model...")
    tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(phi_tot, psi_tot, B, [0,0,1,0,0,0,0,0])
    A_pod, H_pod = tensors_pod
    if rank == 0:
        np.save('results/A_pod.npy', A_pod)
        np.save('results/H_pod.npy', H_pod)

    if not use_ic:
        phi_ic = phi_pod.copy()
        psi_ic = psi_pod.copy()
        A_ic = A_pod.copy()
        H_ic = H_pod.copy()


St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrr = manifolds.Euclidean(r,r,r)
M = manifolds.Product([Gr,St,Euc_rr,Euc_rrr])
M_gasnitrom = manifolds.Product([Gr,St,Euc_rr,Euc_rr,Euc_rr,Euc_rrr])

cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)


## Compute OpInf model
if run_opinf:
    if rank == 0:
        print("\nComputing OpInf Model...")
    weights = pool.weights.copy()
    pool.weights *= pool.n_traj*pool.n_snapshots

    lam = np.logspace(-4,-1,num=30)
    cost_oi = []
    for (count,l) in enumerate(lam):
        tensors_opinf = opinf_fun.operator_inference(pool, phi_pod, poly_comp, [0.0,l])
        point = (phi_pod, psi_pod) + tensors_opinf
        cost_oi.append(cost(*point))
        if rank == 0:
            print(f"  Lambda {count+1}/{len(lam)}: {l:.4e}, Cost: {cost_oi[-1]:.6e}")
        
    pool.weights = weights

    lambdas = [0.0,lam[np.argmin(cost_oi)]]
    if rank == 0:
        print("  Best Lambdas:", lambdas, "Cost:", np.min(cost_oi))

    weights = pool.weights.copy()
    pool.weights *= pool.n_traj*pool.n_snapshots

    t1 = tlib.perf_counter()
    tensors_oi = opinf_fun.operator_inference(pool, phi_pod, poly_comp, lambdas)
    t2 = tlib.perf_counter()
    opinf_time = t2 - t1
    A_oi, H_oi = tensors_oi

    pool.weights = weights

    if rank == 0:
        np.save('results/A_oi.npy', A_oi)
        np.save('results/H_oi.npy', H_oi)
        np.save('results/opinf_time.npy', opinf_time)


## Compute globally stable OpInf model
if run_opinf_gs:
    if rank == 0:
        print("\nTraining OpInf (GS) Model...")
    initial_guess = create_initial_guess(A_ic, H_ic)
    M_opinf = manifolds.Product([Euc_rr,Euc_rr,Euc_rr,Euc_rrr])
    line_searcher = myAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.85,max_iterations=25,initial_step_size=1)
    optimizer = optimizers.ConjugateGradient(max_iterations=2000,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=0)

    weights = pool.weights.copy()
    pool.weights *= pool.n_traj*pool.n_snapshots

    lam = np.logspace(-6,-3,num=30)
    cost_oi_gs = []
    for count, l in enumerate(lam):
        opinf_kwargs = {'glob_stable':True,'regularization_H':l}
        cost_oi_fun, grad_oi_fun = opinf_fun_grad.create_objective_and_gradient(M_opinf,opt_obj,phi_pod,**opinf_kwargs)
        problem = pymanopt.Problem(M_opinf,cost_oi_fun,euclidean_gradient=grad_oi_fun)
        
        result = optimizer.run(problem,initial_point=initial_guess)
        
        cost_func_nit, _, _ = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom,glob_stable=True)
        cost_nit = cost_func_nit(phi_pod,psi_pod,*result.point)
        cost_oi_gs.append(cost_nit)
        if rank == 0:
            print(f"  Lambda {count+1}/{len(lam)}: {l:.4e}, Cost: {cost_oi_gs[-1]:.6e}")

    pool.weights = weights

    lambdas = [0.0,lam[np.argmin(cost_oi_gs)]]
    if rank == 0:
        print("  Best Lambdas:", lambdas, "Cost:", np.min(cost_oi_gs))

    weights = pool.weights.copy()
    pool.weights *= pool.n_traj*pool.n_snapshots

    opinf_kwargs = {'glob_stable':True,'regularization_H':lambdas[1]}
    cost_oi_fun, grad_oi_fun = opinf_fun_grad.create_objective_and_gradient(M_opinf,opt_obj,phi_pod,**opinf_kwargs)
    problem = pymanopt.Problem(M_opinf,cost_oi_fun,euclidean_gradient=grad_oi_fun)
    t1 = tlib.perf_counter()
    result = optimizer.run(problem,initial_point=initial_guess)
    t2 = tlib.perf_counter()
    gasopinf_time = t2-t1

    itervec_gasopinf = result.log["iterations"]["iteration"]
    costvec_gasopinf = result.log["iterations"]["cost"]

    pool.weights = weights

    Qhat, Jhat, Rhat, Hhat = result.point
    A_oi_gs, H_oi_gs = construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]

    if rank == 0:
        np.save('results/Qhat_oi_gs.npy', Qhat)
        np.save('results/Jhat_oi_gs.npy', Jhat)
        np.save('results/Rhat_oi_gs.npy', Rhat)
        np.save('results/Hhat_oi_gs.npy', Hhat)

        np.save('results/A_oi_gs.npy', A_oi_gs)
        np.save('results/H_oi_gs.npy', H_oi_gs)
        np.save('results/gasopinf_time.npy', gasopinf_time)
        np.save('results/itervec_gasopinf.npy', itervec_gasopinf)
        np.save('results/costvec_gasopinf.npy', costvec_gasopinf)


## Compute NiTROM model
if run_nitrom:
    if rank == 0:
        print("\nTraining NiTROM Model...")
    init_point = (phi_ic, psi_ic, A_ic, H_ic)
    times = 10 * np.arange(1, 17, 1)
    all_iters_nit = []
    all_costs_nit = []
    t1 = tlib.perf_counter()
    for i, factor in enumerate(times):
        if rank == 0:
            print(f"\n  Training with factor {factor}, number {i+1}/{len(times)}")

        kouter = 50
        for k in range(kouter):
            if k % 2 == 0:
                which_fix = 'fix_bases'
                max_iter = 5
            else:
                which_fix = 'fix_tensors'
                max_iter = 5
            
            if rank == 0:
                print(f"    Outer loop {k+1}/{kouter}, option {which_fix}...")

            A = init_point[2]
            evals = np.linalg.eigvals(A)
            max_real_part = np.max(evals.real)
            if rank == 0:
                print(f"      Max real part of eigenvalues of A: {max_real_part:.4e}")

            which_times = np.arange(0, factor, 1)
            opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
            opt_obj_kwargs = {'which_fix': which_fix}
            opt_obj = classes.optimization_objects(*opt_obj_inputs, **opt_obj_kwargs)

            cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
            problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
            
            line_searcher = myAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.1,max_iterations=25,initial_step_size=1)
            optimizer = optimizers.ConjugateGradient(max_iterations=max_iter,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=1,verbosity=verb)
            result = optimizer.run(problem,initial_point=init_point)

            all_iters_nit.append(result.log["iterations"]["iteration"])
            all_costs_nit.append(result.log["iterations"]["cost"])

            phi_nit = result.point[0]
            psi_nit = result.point[1]
            phi_nit = phi_nit @ scipy.linalg.inv(psi_nit.T @ phi_nit)
            A_nit, H_nit = result.point[2:]

            init_point = (phi_nit, psi_nit, A_nit, H_nit)
        if rank == 0:
            np.save('results/phi_nit.npy', phi_nit)
            np.save('results/psi_nit.npy', psi_nit)
            np.save('results/A_nit.npy', A_nit)
            np.save('results/H_nit.npy', H_nit)
    
    t2 = tlib.perf_counter()
    nit_time = t2 - t1
    
    if rank == 0:
        np.save('results/nit_time.npy', nit_time)
        np.save('results/all_iters_nit.npy', all_iters_nit)
        np.save('results/all_costs_nit.npy', all_costs_nit)
    


## Compute globally stable NiTROM model
if run_nitrom_gs:
    if rank == 0:
        print("\nTraining NiTROM (GS) Model...")
    if use_ic:
        init_point_gs = (phi_ic, psi_ic, Qhat_ic, Jhat_ic, Rhat_ic, Hhat_ic)

    times = 10 * np.arange(1, 17, 1)
    all_iters_gasnit = []
    all_costs_gasnit = []
    t1 = tlib.perf_counter()
    for i, factor in enumerate(times):
        if rank == 0:
            print(f"\n  Training with factor {factor}, number {i+1}/{len(times)}")

        kouter = 50
        for k in range(kouter):
            if k % 2 == 0:
                which_fix = 'fix_bases'
                max_iter = 5
            else:
                which_fix = 'fix_tensors'
                max_iter = 5
                
            if rank == 0:
                print(f"    Outer loop {k+1}/{kouter}, option {which_fix}...")

            which_times = np.arange(0, factor, 1)
            opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,poly_comp)
            opt_obj_kwargs = {'which_fix': which_fix}
            opt_obj = classes.optimization_objects(*opt_obj_inputs, **opt_obj_kwargs)

            nitrom_kwargs = {'glob_stable':True}
            cost, grad, hess = nitrom_functions.create_objective_and_gradient(M_gasnitrom,opt_obj,pool,fom,**nitrom_kwargs)
            problem = pymanopt.Problem(M_gasnitrom,cost,euclidean_gradient=grad)
            
            line_searcher = myAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.1,max_iterations=25,initial_step_size=1)
            optimizer = optimizers.ConjugateGradient(max_iterations=max_iter,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=1,verbosity=verb)

            result = optimizer.run(problem,initial_point=init_point_gs)
            phi_nit_gs = result.point[0]
            psi_nit_gs = result.point[1]
            phi_nit_gs = phi_nit_gs @ scipy.linalg.inv(psi_nit_gs.T @ phi_nit_gs)
            Qhat, Jhat, Rhat, Hhat = result.point[2:]
            A_nit_gs, H_nit_gs = construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]

            all_iters_gasnit.append(result.log["iterations"]["iteration"])
            all_costs_gasnit.append(result.log["iterations"]["cost"])

            init_point_gs = (phi_nit_gs, psi_nit_gs, Qhat, Jhat, Rhat, Hhat)

            if rank == 0:
                np.save('results/phi_nit_gs.npy', phi_nit_gs)
                np.save('results/psi_nit_gs.npy', psi_nit_gs)
                np.save('results/A_nit_gs.npy', A_nit_gs)
                np.save('results/H_nit_gs.npy', H_nit_gs)
                np.save('results/Qhat_nit_gs.npy', Qhat)
                np.save('results/Jhat_nit_gs.npy', Jhat)
                np.save('results/Rhat_nit_gs.npy', Rhat)
                np.save('results/Hhat_nit_gs.npy', Hhat)



    t2 = tlib.perf_counter()
    gasnit_time = t2-t1
    if rank == 0:
        np.save('results/gasnit_time.npy', gasnit_time)
        np.save('results/all_iters_gasnit.npy', all_iters_gasnit)
        np.save('results/all_costs_gasnit.npy', all_costs_gasnit)