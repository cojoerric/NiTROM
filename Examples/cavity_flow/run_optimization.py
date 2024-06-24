
import os
import numpy as np 
import scipy
import torch 
import time as tlib

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from pymanopt.optimizers.line_search import AlbysAdaptiveLineSearcher, AdaptiveLineSearcher

import optimization_functions as opt_functions

path = "./data/"

print("----------------------------------------------------------------------")
print("-------- Is GPU available? -------------------------------------------")
print(torch.cuda.is_available())
device = torch.device(torch.cuda.current_device())
print("----------------------------------------------------------------------")


print("----------------------------------------------------------------------")
print("----- Loading data and defining the C matrix -------------------------")

B = np.loadtxt(path + "training_B_proj.txt")
Data = np.loadtxt(path + "training_data_proj.txt")
Amps = np.loadtxt(path + "training_Amps.txt")
tsave = np.loadtxt(path + "training_tsave.txt")

try: 
    ntraj = len(Amps)
except:
    Amps = [Amps]
    ntraj = len(Amps)


nsnaps = Data.shape[-1]//ntraj 
Data_ = np.zeros((ntraj,Data.shape[0],nsnaps))
weights = np.zeros(ntraj)
for k in range (ntraj):
    Data_[k,] = Data[:,k*nsnaps:(k+1)*nsnaps]
    weights[k] = np.mean(np.linalg.norm(Data_[k,],axis=0)**2)

Data_ = torch.from_numpy(Data_).to(device)
B_ = torch.from_numpy(B).to(device)
print(weights)

weights *= ntraj*nsnaps
# weights = np.ones(ntraj)*ntraj*nsnaps

N = Data.shape[0]
# N = 99550

C = np.ones(N)
rows = torch.tensor(np.arange(0,len(C)+1,1),dtype=torch.int64)
cols = torch.tensor(np.arange(0,len(C),1),dtype=torch.int64)
CMat = torch.sparse_csc_tensor(rows,cols,torch.tensor(C),size=(len(C),len(C)))#.to(device)

vec = np.random.randn(N)
vec = vec/np.linalg.norm(vec)
vec = torch.from_numpy(vec)
print(np.linalg.norm((vec - CMat@vec).numpy()))

CMat = CMat.to(device)

print("----------------------------------------------------------------------")

print("----------------------------------------------------------------------")
print("-------- Load Phi and A ----------------------------------------------")

# Phi = np.loadtxt(path + "training_Phi.txt")
# Psi = np.loadtxt(path + "training_Psi.txt")
# r = Phi.shape[-1]
# A = np.loadtxt(path + "training_A.txt")
# H = np.loadtxt(path + "training_H.txt").reshape((r,r,r))

# print("Model size = %d"%r)

# print(np.linalg.norm(A),np.linalg.norm(H))

Phi = np.loadtxt(path + "Phi_opt.txt")
Psi = np.loadtxt(path + "Psi_opt.txt")
r = Phi.shape[-1]
A = np.loadtxt(path + "A_opt.txt")
H = np.loadtxt(path + "H_opt.txt").reshape((r,r,r))

r = Phi.shape[-1]
print("ROM size = %d"%r)

ev, evecs = scipy.linalg.eig(A)
idces = np.argwhere(ev.real > 0).reshape(-1)
print(ev[idces])
print(len(tsave))
ev[idces] = -0.001 + 1j*ev[idces].imag
A = (evecs@np.diag(ev)@scipy.linalg.inv(evecs)).real


print("----------------------------------------------------------------------")


pen = 1e-3 
tsp = 100 
ic = np.random.randn(r)
ic = ic/np.linalg.norm(ic)

opt_obj = opt_functions.optimization_objects(CMat,Data_[:,:,:100],tsave[:100],weights[:],5,\
                                             (tsave[1]-tsave[0])/15,device,'both',pen,tsp,ic)

print("Size of Q = (%d,%d,%d)"%(opt_obj.X_train.shape))
print("Size of time = (%d)"%len(opt_obj.time_train))
print("Size of weights = (%d)"%len(opt_obj.weights))
print(opt_obj.weights)


# Phi = torch.from_numpy(Phi)
# Psi = torch.from_numpy(Psi)
# A = torch.from_numpy(A)
# H = torch.from_numpy(H)
# L = torch.from_numpy(L)

# t0 = tlib.time()
# J = opt_functions.cost(Phi,Phi,A,H,L,opt_obj)
# tf = tlib.time()
# print("Time to evaluate the cost function = %1.12e"%(tf - t0))

# t0 = tlib.time()
# J = opt_functions.euclidean_gradient(Phi,Psi,A,H,L,opt_obj)
# tf = tlib.time()
# print("Time to evaluate the gradient = %1.12e"%(tf - t0))

# opt_functions.test_gradients(3e-8,Phi,Psi,A,H,L,opt_obj)

n, r = Phi.shape

St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrr = manifolds.Euclidean(r,r,r)

M = manifolds.Product([Gr,St,Euc_rr,Euc_rrr])
line_searcher = AdaptiveLineSearcher(contraction_factor=0.4,sufficient_decrease=0.2,max_iterations=20,initial_step_size=1)

point = [Phi,Psi,A,H]
n_outer = 50

for jj in range (1):

    n_times = 150 + jj*50
    ww = np.zeros(Data_.shape[0])
    for j in range (len(ww)):
        ww[j] = np.mean(np.linalg.norm(Data_[j,:,:n_times].cpu(),axis=0)**2)*ntraj*n_times
    print(ww)

    for k in range (n_outer):

        if np.mod(k,2) == 0:    which = 'just_tensors'
        else:                   which = 'just_bases'
        # which = 'just_tensors'

        opt_obj = opt_functions.optimization_objects(CMat,Data_[:,:,:n_times],tsave[:n_times],ww[:],\
                                                     10,(tsave[1]-tsave[0])/15,device,which,pen,tsp,ic)
        cost, grad, hess = opt_functions.create_objective_and_gradient_parallel(M,opt_obj)
        problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)

        print("Running outer iteration number %d/%d with n_times = %d. Flag: %s"%(k+1,n_outer,n_times,opt_obj.which))

        optimizer = optimizers.ConjugateGradient(max_iterations=5,min_step_size=1e-20,max_time=3600*1.7,line_searcher=line_searcher) 
        result = optimizer.run(problem,initial_point=point,reuse_line_searcher=True)
        point = result.point

        if np.mod(k,5) == 0: 

            print("Saving data...")
            Phi_opt = point[0]
            Psi_opt = point[1]
            A_opt = point[2]
            H_opt = point[3]

            np.savetxt(path + "Phi_opt.txt",Phi_opt)
            np.savetxt(path + "Psi_opt.txt",Psi_opt)
            np.savetxt(path + "A_opt.txt",A_opt)
            np.savetxt(path + "H_opt.txt",H_opt.reshape((r,r**2)))

    Phi_opt = point[0]
    Psi_opt = point[1]
    A_opt = point[2]
    H_opt = point[3]

    np.savetxt(path + "Phi_opt.txt",Phi_opt)
    np.savetxt(path + "Psi_opt.txt",Psi_opt)
    np.savetxt(path + "A_opt.txt",A_opt)
    np.savetxt(path + "H_opt.txt",H_opt.reshape((r,r**2)))


