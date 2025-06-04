import numpy as np 
import scipy as sp
import math
from string import ascii_lowercase as ascii


def compute_indices(c_ls=[], c=0, idx=5, r=5, order=0):
    
    for i in range(idx):
        ci = c + i * r**order
        if (order == 0):    c_ls.append(ci)
        else:               c_ls = compute_indices(c_ls,ci,i+1,r,order-1)
    return c_ls

        
def perform_POD(mpi_pool,r):
    
    N = mpi_pool.n_snapshots*mpi_pool.n_traj
    X = np.zeros((mpi_pool.X.shape[1],N))
    
    for i in range (mpi_pool.n_traj):
        X[:,i*mpi_pool.n_snapshots:(i+1)*mpi_pool.n_snapshots] = mpi_pool.X[i,]
        
    u, s, _ = sp.linalg.svd(X,full_matrices=False)
    
    return u[:,:r], (100*np.cumsum(s[:r]**2)/np.sum(s**2))[-1]

def assemble_Y(mpi_pool,Phi):
    
    r = Phi.shape[-1]
    Y = np.zeros((r,mpi_pool.n_traj*mpi_pool.n_snapshots))
    for i in range (mpi_pool.n_traj):
        Y[:,i*mpi_pool.n_snapshots:(i+1)*mpi_pool.n_snapshots] = Phi.T@mpi_pool.dX[i,]
    
    return Y

def assemble_W(mpi_pool):
    
    W = np.zeros(mpi_pool.n_traj*mpi_pool.n_snapshots)
    for i in range (mpi_pool.n_traj):
        W[i*mpi_pool.n_snapshots:(i+1)*mpi_pool.n_snapshots] = 1./(mpi_pool.weights[i]*mpi_pool.n_traj)
    
    return np.diag(W)


def assemble_Z(mpi_pool,Phi,poly_comp):
    
    r = Phi.shape[-1]
    
    for (count,p) in enumerate(poly_comp):
        
        rp = math.comb(r+p-1,p)
        idces = compute_indices([],0,r,r,p-1)
        equation = ','.join(ascii[:p])
        Z_ = np.zeros((rp,mpi_pool.n_traj*mpi_pool.n_snapshots))
        for i in range (mpi_pool.n_traj):
            for j in range (mpi_pool.n_snapshots):
                idx = i*mpi_pool.n_snapshots + j
                operands = [Phi.T@mpi_pool.X[i,:,j] for _ in range (p)]
                Z_[:,idx] = (np.einsum(equation,*operands).reshape(-1))[idces]
        if count == 0:  Z = Z_.copy()
        else:           Z = np.concatenate((Z,Z_),axis=0)


    return Z

def assemble_P(r,poly_comp,lambdas):
    
    for (count,p) in enumerate(poly_comp):
        
        rp = math.comb(r+p-1,p)
        P_ = lambdas[count]*np.ones(rp)
        
        if count == 0:  P = P_.copy()
        else:           P = np.concatenate((P,P_))
    
    return np.diag(P)


def extract_tensors(r,poly_comp,S):
    
    tensors = []
    shift = 0
    for p in poly_comp:
        
        rp = math.comb(r+p-1,p)
        idces = compute_indices([],0,r,r,p-1)
        
        T = np.zeros((r,r**p))
        T[:,idces] = S[:,shift:shift+rp]
        reshape_list = [r for _ in range (p+1)]
        tensors.append(T.reshape(*reshape_list))
        
        shift += rp
    
    return tuple(tensors)
    
        
def solve_least_squares_problem(mpi_pool,Z,Y,W,P):
    
    """
        Solves the weighted least squares problem with L2 regularization. 
        The solution is M = Y@W@Z.T@inv(Z@W@Z.T + P)
    """

    u, s, _ = sp.linalg.svd(Z@W@Z.T + P)
    idces = np.argwhere(s > 1e-12).reshape(-1)
    
    return Y@W@Z.T@u[:,idces]@np.diag(1./s[idces])@(u[:,idces]).T



def operator_inference(mpi_pool,Phi,poly_comp,lambdas):
    
    n, r = Phi.shape
    
    W = assemble_W(mpi_pool)
    Y = assemble_Y(mpi_pool,Phi)
    Z = assemble_Z(mpi_pool,Phi,poly_comp)
    print(Y.shape)
    P = assemble_P(r,poly_comp,lambdas)
    
    S = solve_least_squares_problem(mpi_pool,Z,Y,W,P)
    
    return extract_tensors(r,poly_comp,S)
    
  
