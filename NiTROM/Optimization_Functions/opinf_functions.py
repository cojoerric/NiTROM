import torch
from .utils import construct_operators, propagate_gradients


def create_objective_and_gradient(*args, **kwargs):

    opt_obj, phi = args
    glob_stable = kwargs.get('glob_stable', False)
    poly_comp = kwargs.get('poly_comp', None)
    return_numpy = kwargs.get('return_numpy', False)
    regularization_H = kwargs.get('regularization_H', 0.0)

    B = opt_obj.my_n_traj
    n, r = phi.shape
    T = opt_obj.n_snapshots

    Z = torch.zeros((B, r, T), device=phi.device, dtype=phi.dtype)
    dZ = torch.zeros_like(Z)
    W = torch.zeros(B*T, device=phi.device, dtype=phi.dtype)
    for k in range(B):
        Z[k] = phi.T @ opt_obj.X[k]
        dZ[k] = phi.T @ opt_obj.dX[k]
        W[k*T:(k+1)*T] = 1/(opt_obj.weights[k] * opt_obj.my_n_traj)
    W = torch.diag(W)

    def cost_fn(*params):
        if glob_stable:
            (A, H), _ = construct_operators(params, poly_comp)
        else:
            A, H = params
        linear_term = torch.matmul(A, Z)
        quadratic_term = torch.einsum('ijk,bjt,bkt->bit', H, Z, Z)
        R = dZ - (linear_term + quadratic_term)
        R_flat = R.permute(1, 0, 2).reshape(r, B*T)
        loss = ((R_flat @ W) * R_flat).sum() + regularization_H * torch.norm(H)**2

        if return_numpy:
            loss = loss.cpu().numpy()

        return loss
    
    def grad_fn(*params):
        if glob_stable:
            tensors_new, other_tensors = construct_operators(params, poly_comp)
        else:
            tensors_new = params

        A, H = tensors_new
        linear_term = torch.matmul(A, Z)
        quadratic_term = torch.einsum('ijk,bjt,bkt->bit', H, Z, Z)
        R = dZ - (linear_term + quadratic_term)
        R_flat = R.permute(1, 0, 2).reshape(r, B*T)
        RW_flat = R_flat @ W
        RW = RW_flat.reshape(r, B, T).permute(1, 0, 2)
        
        grad_A = -2 * torch.einsum('bit,bjt->ij', RW, Z)
        grad_H = -2 * torch.einsum('bit,bjt,bkt->ijk', RW, Z, Z)
        grad_H += 2 * regularization_H * H
        grads = (grad_A, grad_H)

        if glob_stable:
            grads_new = propagate_gradients(grads, other_tensors, params, poly_comp)
        else:
            grads_new = grads

        if return_numpy:
            grads_new = tuple(grad.cpu().numpy() for grad in grads_new)
            return grads_new
        else:
            return grads_new
    
    return cost_fn, grad_fn