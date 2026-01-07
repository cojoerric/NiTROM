import torch
import math
from string import ascii_lowercase as ascii


def _get_ref_tensor(*arrays):
    for array in arrays:
        if torch.is_tensor(array):
            return array
    return None


def _as_tensor(array, ref=None):
    if torch.is_tensor(array):
        if ref is not None and (array.device != ref.device or array.dtype != ref.dtype):
            return array.to(device=ref.device, dtype=ref.dtype)
        return array
    tensor = torch.as_tensor(array)
    if ref is not None:
        tensor = tensor.to(device=ref.device, dtype=ref.dtype)
    return tensor


def compute_indices(c_ls=[], c=0, idx=5, r=5, order=0):
    
    for i in range(idx):
        ci = c + i * r**order
        if (order == 0):    c_ls.append(ci)
        else:               c_ls = compute_indices(c_ls,ci,i+1,r,order-1)
    return c_ls

        
def perform_POD(pool,r):
    
    ref = _get_ref_tensor(pool.X)
    N = pool.n_snapshots*pool.n_traj
    X = torch.zeros((pool.X.shape[1],N), device=ref.device if ref is not None else None, dtype=ref.dtype if ref is not None else None)
    X_full = _as_tensor(pool.X, ref=ref)
    
    for i in range (pool.n_traj):
        X[:,i*pool.n_snapshots:(i+1)*pool.n_snapshots] = X_full[i,]
        
    u, s, _ = torch.linalg.svd(X,full_matrices=False)
    
    return u[:,:r], (100*torch.cumsum(s[:r]**2, dim=0)/torch.sum(s**2))[-1]

def assemble_Y(pool,Phi):
    
    ref = _get_ref_tensor(Phi, pool.dX)
    Phi = _as_tensor(Phi, ref=ref)
    dX = _as_tensor(pool.dX, ref=ref)
    r = Phi.shape[-1]
    Y = torch.zeros((r,pool.n_traj*pool.n_snapshots), device=Phi.device, dtype=Phi.dtype)
    for i in range (pool.n_traj):
        Y[:,i*pool.n_snapshots:(i+1)*pool.n_snapshots] = Phi.T@dX[i,]
    
    return Y

def assemble_W(pool):
    
    ref = _get_ref_tensor(pool.weights)
    weights = _as_tensor(pool.weights, ref=ref)
    W = torch.zeros(pool.n_traj*pool.n_snapshots, device=weights.device, dtype=weights.dtype)
    for i in range (pool.n_traj):
        W[i*pool.n_snapshots:(i+1)*pool.n_snapshots] = 1./(weights[i]*pool.n_traj)
    
    return torch.diag(W)


def assemble_Z(pool,Phi,poly_comp):
    
    ref = _get_ref_tensor(Phi, pool.X)
    Phi = _as_tensor(Phi, ref=ref)
    X = _as_tensor(pool.X, ref=ref)
    r = Phi.shape[-1]
    
    for (count,p) in enumerate(poly_comp):
        
        rp = math.comb(r+p-1,p)
        idces = compute_indices([],0,r,r,p-1)
        equation = ','.join(ascii[:p])
        Z_ = torch.zeros((rp,pool.n_traj*pool.n_snapshots), device=Phi.device, dtype=Phi.dtype)
        idces_t = torch.as_tensor(idces, device=Phi.device, dtype=torch.long)
        for i in range (pool.n_traj):
            for j in range (pool.n_snapshots):
                idx = i*pool.n_snapshots + j
                operands = [Phi.T@X[i,:,j] for _ in range (p)]
                Z_[:,idx] = (torch.einsum(equation,*operands).reshape(-1))[idces_t]
        if count == 0:  Z = Z_.clone()
        else:           Z = torch.cat((Z,Z_),dim=0)


    return Z

def assemble_P(r,poly_comp,lambdas):
    
    ref = _get_ref_tensor(lambdas)
    lambdas = _as_tensor(lambdas, ref=ref)
    for (count,p) in enumerate(poly_comp):
        
        rp = math.comb(r+p-1,p)
        P_ = lambdas[count]*torch.ones(rp, device=lambdas.device, dtype=lambdas.dtype)
        
        if count == 0:  P = P_.clone()
        else:           P = torch.cat((P,P_))
    
    return torch.diag(P)


def extract_tensors(r,poly_comp,S):
    
    ref = _get_ref_tensor(S)
    S = _as_tensor(S, ref=ref)
    tensors = []
    shift = 0
    for p in poly_comp:
        
        rp = math.comb(r+p-1,p)
        idces = compute_indices([],0,r,r,p-1)
        idces_t = torch.as_tensor(idces, device=S.device, dtype=torch.long)
        T = torch.zeros((r,r**p), device=S.device, dtype=S.dtype)
        T[:,idces_t] = S[:,shift:shift+rp]
        reshape_list = [r for _ in range (p+1)]
        tensors.append(T.reshape(*reshape_list))
        
        shift += rp
    
    return tuple(tensors)
    
        
def solve_least_squares_problem(pool,Z,Y,W,P):
    
    """
        Solves the weighted least squares problem with L2 regularization. 
        The solution is M = Y@W@Z.T@inv(Z@W@Z.T + P)
    """

    u, s, _ = torch.linalg.svd(Z@W@Z.T + P, full_matrices=False)
    idces = torch.where(s > 1e-12)[0]
    
    return Y@W@Z.T@u[:,idces]@torch.diag(1./s[idces])@(u[:,idces]).T



def operator_inference(pool,Phi,poly_comp,lambdas):
    
    ref = _get_ref_tensor(Phi)
    Phi = _as_tensor(Phi, ref=ref)
    n, r = Phi.shape
    
    W = _as_tensor(assemble_W(pool), ref=Phi)
    Y = assemble_Y(pool,Phi)
    Z = assemble_Z(pool,Phi,poly_comp)
    P = _as_tensor(assemble_P(r,poly_comp,lambdas), ref=Phi)
    
    S = solve_least_squares_problem(pool,Z,Y,W,P)
    
    return extract_tensors(r,poly_comp,S)
    
  
