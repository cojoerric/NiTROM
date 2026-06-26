import numpy as np 
import scipy 
from mpi4py import MPI
import time as tlib

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from pymanopt.tools.diagnostics import check_gradient

from NiTROM.Optimization_Functions import classes, nitrom_functions, opinf_functions as opinf_fun, opinf_functions_grad as opinf_fun_grad
from NiTROM.Optimization_Functions.utils import create_initial_guess, construct_operators
from NiTROM.PyManopt_Functions.my_pymanopt_classes import myAdaptiveLineSearcher
import fom_class


cPOD, cOI, cTR, cOPT = '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'
lPOD, lOI, lTR, lOPT = 'solid', 'dotted', 'dashed', 'dashdot'

run_nitrom = 0
run_opinf = 0
run_gasopinf = 1
run_gasnitrom = 0


# Instantiate the full-order model class

n = 3 
beta = 20.0
A2 = np.diag([-1,-2,-5])
A3 = np.zeros((3,3,3))
A3[:,:,-1] = np.diag([beta,beta,0.0])
B = np.ones((3,1))
C = np.ones((1,3))

fom = fom_class.full_order_model(A2,A3,B,C)


# Load training trajectories

max_val = 5/20
traj_path = "./trajectories/"

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

n_traj = 4


# Compute POD model 

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_steady_forcing':fname_forcing,'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)

r = 2               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part

Phi_pod, _ = opinf_fun.perform_POD(pool,2)
Psi_pod = Phi_pod.copy()
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi_pod,Psi_pod)
A_pod, H_pod = tensors_pod

np.save('results/phi_pod.npy',Phi_pod)
np.save('results/psi_pod.npy',Psi_pod)
np.save('results/A_pod.npy',A_pod)
np.save('results/H_pod.npy',H_pod)


# Compute NiTROM model 

which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 2

opt_obj_inputs = (pool,which_trajs,which_times,leggauss_deg,nsave_rom,[1,2])
opt_obj = classes.optimization_objects(*opt_obj_inputs)
St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrr = manifolds.Euclidean(r,r,r)
M = manifolds.Product([Gr,St,Euc_rr,Euc_rrr])

if run_nitrom:

    nitrom_args = (M,opt_obj,pool,fom)
    cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
    problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
    # check_gradient(problem,x=[Phi_pod,Psi_pod,*tensors_pod])

    line_searcher = myAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.85,max_iterations=25,initial_step_size=1)
    optimizer = optimizers.ConjugateGradient(max_iterations=3500,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=1)

    point = (Phi_pod,Psi_pod) + tensors_pod
    t1 = tlib.perf_counter()
    result = optimizer.run(problem,initial_point=point)
    t2 = tlib.perf_counter()
    nit_time = t2-t1

    Phi_nit = result.point[0]
    Psi_nit = result.point[1]
    Phi_nit = Phi_nit@scipy.linalg.inv(Psi_nit.T@Phi_nit)
    tensors_nit = tuple(result.point[2:])
    A_nit, H_nit = tensors_nit

    itervec_nit = result.log["iterations"]["iteration"]
    costvec_nit = result.log["iterations"]["cost"]
    gradvec_nit = result.log["iterations"]["gradient_norm"]

    np.save('results/phi_nit.npy',Phi_nit)
    np.save('results/psi_nit.npy',Psi_nit)
    np.save('results/A_nit.npy',A_nit)
    np.save('results/H_nit.npy',H_nit)
    np.save('results/itervec_nit.npy',itervec_nit)
    np.save('results/costvec_nit.npy',costvec_nit)
    np.save('results/nitrom_time.npy',nit_time)


# Compute OpInf model

if run_opinf:
    weights = pool.weights.copy()
    pool.weights *= pool.n_traj*pool.n_snapshots

    lam = np.logspace(-8,-2,num=100)
    cost_oi = []
    for (count,l) in enumerate(lam):
        tensors_opinf = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,[0.0,l])
        point = (Phi_pod,Psi_pod) + tensors_opinf
        cost_oi.append(cost(*point))

    pool.weights = weights

    lambdas = [0.0,lam[np.argmin(cost_oi)]]
    print(np.min(cost_oi),lambdas)
    weights = pool.weights.copy()
    pool.weights *= pool.n_traj*pool.n_snapshots
    t1 = tlib.perf_counter()
    tensors_oi = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,lambdas)
    t2 = tlib.perf_counter()
    opinf_time = t2-t1
    pool.weights = weights
    A_oi, H_oi = tensors_oi

    np.save('results/A_oi.npy',A_oi)
    np.save('results/H_oi.npy',H_oi)
    np.save('results/opinf_time.npy',opinf_time)


# Compute GasOpInf model

if run_gasopinf:
    initial_guess = create_initial_guess(A_pod, H_pod)
    M_opinf = manifolds.Product([Euc_rr,Euc_rr,Euc_rr,Euc_rrr])
    line_searcher = myAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.85,max_iterations=25,initial_step_size=1)
    optimizer = optimizers.ConjugateGradient(max_iterations=2000,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,verbosity=1,log_verbosity=1)

    weights = pool.weights.copy()
    pool.weights *= pool.n_traj*pool.n_snapshots
    lam = np.logspace(-8,-7,num=15)
    cost_oi_gs = []
    for (count,l) in enumerate(lam):
        opinf_kwargs = {'glob_stable':True,'regularization_H':l}
        cost, grad = opinf_fun_grad.create_objective_and_gradient(M_opinf,opt_obj,Phi_pod,**opinf_kwargs)
        problem = pymanopt.Problem(M_opinf,cost,euclidean_gradient=grad)
        result = optimizer.run(problem,initial_point=initial_guess)
        
        cost_func_nit, _, _ = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom,glob_stable=True)
        cost_nit = cost_func_nit(Phi_pod,Psi_pod,*result.point)
        cost_oi_gs.append(cost_nit)

    pool.weights = weights
    lambda_gs = lam[np.argmin(cost_oi_gs)]
    print(np.min(cost_oi_gs), lambda_gs)
    weights = pool.weights.copy()
    pool.weights *= pool.n_traj*pool.n_snapshots
    opinf_kwargs = {'glob_stable':True,'regularization_H':lambda_gs}
    cost, grad = opinf_fun_grad.create_objective_and_gradient(M_opinf,opt_obj,Phi_pod,**opinf_kwargs)
    problem = pymanopt.Problem(M_opinf,cost,euclidean_gradient=grad)
    optimizer = optimizers.ConjugateGradient(max_iterations=3500,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,verbosity=1,log_verbosity=1)
    t1 = tlib.perf_counter()
    result = optimizer.run(problem,initial_point=initial_guess)
    t2 = tlib.perf_counter()
    pool.weights = weights
    gasopinf_time = t2-t1
    Qhat, Jhat, Rhat, Hhat = result.point
    A_oi_gs, H_oi_gs = construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]

    itervec_oi_gs = result.log["iterations"]["iteration"]
    costvec_oi_gs = result.log["iterations"]["cost"]

    np.save('results/A_oi_gs.npy',A_oi_gs)
    np.save('results/H_oi_gs.npy',H_oi_gs)
    np.save('results/itervec_oi_gs.npy',itervec_oi_gs)
    np.save('results/costvec_oi_gs.npy',costvec_oi_gs)
    np.save('results/gasopinf_time.npy',gasopinf_time)


# Compute GasNiTROM model

if run_gasnitrom:
    line_searcher = myAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.05,max_iterations=10,initial_step_size=1)
    optimizer = optimizers.ConjugateGradient(max_iterations=3500,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=1)
    M_gasnitrom = manifolds.Product([Gr,St,Euc_rr,Euc_rr,Euc_rr,Euc_rrr])

    point = (Phi_pod,Psi_pod) + (Qhat, Jhat, Rhat, Hhat)
    nitrom_kwargs = {'glob_stable':True}
    cost, grad, hess = nitrom_functions.create_objective_and_gradient(M_gasnitrom,opt_obj,pool,fom,**nitrom_kwargs)
    problem = pymanopt.Problem(M_gasnitrom,cost,euclidean_gradient=grad)
    t1 = tlib.perf_counter()
    result = optimizer.run(problem,initial_point=point)
    t2 = tlib.perf_counter()
    gasnit_time = t2-t1

    Phi_nit_gs = result.point[0]
    Psi_nit_gs = result.point[1]
    Phi_nit_gs = Phi_nit_gs@scipy.linalg.inv(Psi_nit_gs.T@Phi_nit_gs)
    Qhat, Jhat, Rhat, Hhat = result.point[2:]
    A_nit_gs, H_nit_gs = construct_operators((Qhat, Jhat, Rhat, Hhat), poly_comp)[0]

    itervec_nit = result.log["iterations"]["iteration"]
    costvec_nit = result.log["iterations"]["cost"]

    np.save('results/phi_nit_gs.npy',Phi_nit_gs)
    np.save('results/psi_nit_gs.npy',Psi_nit_gs)
    np.save('results/A_nit_gs.npy',A_nit_gs)
    np.save('results/H_nit_gs.npy',H_nit_gs)
    np.save('results/itervec_nit.npy',itervec_nit)
    np.save('results/costvec_nit.npy',costvec_nit)
    np.save('results/gasnitrom_time.npy',gasnit_time)