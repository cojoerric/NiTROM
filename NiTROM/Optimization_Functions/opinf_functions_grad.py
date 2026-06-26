import numpy as np
import pymanopt

from .utils import construct_operators, propagate_gradients


def create_objective_and_gradient(*args, **kwargs):

    manifold, opt_obj, phi = args
    poly_comp = opt_obj.poly_comp
    glob_stable = kwargs.get('glob_stable', False)
    regularization_H = kwargs.get('regularization_H', 0.0)

    B = opt_obj.my_n_traj
    n, r = phi.shape
    T = opt_obj.n_snapshots

    Z = np.zeros((B, r, T))
    dZ = np.zeros_like(Z)
    W = np.zeros(B*T)
    for k in range(B):
        Z[k] = phi.T @ opt_obj.X[k]
        dZ[k] = phi.T @ opt_obj.dX[k]
        W[k*T:(k+1)*T] = 1/(opt_obj.weights[k] * opt_obj.my_n_traj)
    W = np.diag(W)

    @pymanopt.function.numpy(manifold)
    def cost(*params):
        if glob_stable:
            (A, H), _ = construct_operators(params, poly_comp)
        else:
            A, H = params
        linear_term = np.matmul(A, Z)
        quadratic_term = np.einsum('ijk,bjt,bkt->bit', H, Z, Z)
        R = dZ - (linear_term + quadratic_term)
        R_flat = R.transpose(1, 0, 2).reshape(r, B*T)
        loss = ((R_flat @ W) * R_flat).sum() + regularization_H * np.linalg.norm(H)**2

        return loss
    
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params):
        if glob_stable:
            tensors_new, other_tensors = construct_operators(params, poly_comp)
        else:
            tensors_new = params

        A, H = tensors_new
        linear_term = np.matmul(A, Z)
        quadratic_term = np.einsum('ijk,bjt,bkt->bit', H, Z, Z)
        R = dZ - (linear_term + quadratic_term)
        R_flat = R.transpose(1, 0, 2).reshape(r, B*T)
        RW_flat = R_flat @ W
        RW = RW_flat.reshape(r, B, T).transpose(1, 0, 2)
        
        grad_A = -2 * np.einsum('bit,bjt->ij', RW, Z)
        grad_H = -2 * np.einsum('bit,bjt,bkt->ijk', RW, Z, Z)
        grad_H += 2 * regularization_H * H
        grads = (grad_A, grad_H)

        if glob_stable:
            grads_new = propagate_gradients(grads, other_tensors, params, poly_comp)
        else:
            grads_new = grads

        return grads_new
    
    return cost, euclidean_gradient