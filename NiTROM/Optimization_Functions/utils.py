import numpy as np
from scipy.linalg import solve_continuous_lyapunov, inv, cholesky


def create_initial_guess(A, H=None, r=None):
    """
    Build initialization tensors for the globally stable model from A (and optional H).
    The construction uses Q = I, J = skew(A), R = -sym(A) projected to PSD, and Hhat = 0.5 * H.
    Note: the resulting H corresponds to the antisymmetric part of the input H.
    """
    if r is None:
        r = A.shape[0]

    I = np.eye(r)

    # Shift spectrum of A into LHP if A is not already stable
    max_eigval = np.max(np.real(np.linalg.eigvals(A)))
    shift = max_eigval + 0.1 if max_eigval > 0 else 0.0
    A = A - shift * I

    # Solve Lyapunov equation for P
    P = solve_continuous_lyapunov(A.T, -I)
    P = 0.5 * (P + P.T)
    Pinv = inv(P)

    # Split A P^{-1} = N - M with N skew and M = -sym(A P^{-1}}) = P^{-2}/2 SPD
    APinv = A @ Pinv
    N = 0.5 * (APinv - APinv.T)
    M = -0.5 * (APinv + APinv.T)

    # J = N / 2; R^{-1} R^{-T} = M; Q^{-1} Q^{-T} = P
    J = 0.5 * N
    R = cholesky(inv(M))
    Q = cholesky(Pinv)

    # S_{:, :, k} = skew(H_{:, :, k})
    S = 0.5 * (H - H.transpose(1, 0, 2))

    # Verify reconstruction A = ((J - J^T) - R^{-1}R^{-T}) Q^{-1} Q^{-T}
    Qinv = inv(Q)
    Rinv = inv(R)
    A_reconstructed = ((J - J.T) - Rinv @ Rinv.T) @ (Qinv @ Qinv.T)
    err = np.linalg.norm(A - A_reconstructed) / np.linalg.norm(A)
    if err > 1e-6:
        raise RuntimeWarning(f"Reconstruction error {err} > 1e-6")
    
    return Q, J, R, S


def compute_Q(Qhat):
    Q_inv = inv(Qhat)
    Q = Q_inv @ Q_inv.T
    return Q, Q_inv


def compute_JR(Jhat, Rhat):
    J = Jhat - Jhat.T
    R_inv = inv(Rhat)
    R = R_inv @ R_inv.T
    return J, R, R_inv


def compute_H(Hhat, Q):
    H2 = np.transpose(Hhat, (2, 1, 0))
    M_tensor = Hhat - H2
    H = np.tensordot(M_tensor, Q, axes=([2], [0]))
    return H


def construct_operators(tensors, poly_comp):
    tensors_new = []
    other_tensors = []
    
    Qhat = tensors[0]
    Q, Q_inv = compute_Q(Qhat)
    other_tensors.extend([Q, Q_inv])

    if 1 in poly_comp:
        Jhat = tensors[1]
        Rhat = tensors[2]
        J, R, R_inv = compute_JR(Jhat, Rhat)
        A = (J - R) @ Q
        tensors_new.append(A)
        other_tensors.extend([J, R, R_inv])
    if 2 in poly_comp:
        Hhat = tensors[-1]
        H = compute_H(Hhat, Q)
        tensors_new.append(H)
    
    return tuple(tensors_new), tuple(other_tensors)


def propagate_gradients(grads, tensors, tensors_hat, poly_comp):
    grad_tensors = [np.zeros_like(tensor_hat) for tensor_hat in tensors_hat]
    Q = tensors[0]; Q_inv = tensors[1]
    grad_Q = np.zeros_like(Q)

    if 1 in poly_comp:
        J = tensors[2]; R = tensors[3]; R_inv = tensors[4]
        grad_A = grads[0]
        grad_J = grad_A @ Q.T; grad_R = -grad_J; grad_Q += (J - R).T @ grad_A
        grad_Jhat = grad_J - grad_J.T; grad_Rhat =  - R_inv.T @ (grad_R + grad_R.T) @ R
        grad_tensors[1] = grad_Jhat; grad_tensors[2] = grad_Rhat
    if 2 in poly_comp:
        Hhat = tensors_hat[-1]
        grad_H = grads[-1]
        S = np.tensordot(grad_H, Q, axes=([2], [0]))
        grad_Hhat = S - np.transpose(S, (2, 1, 0))
        grad_tensors[-1] = grad_Hhat

        G = Hhat - np.transpose(Hhat, (2, 1, 0))
        grad_Q += np.tensordot(grad_H, G, axes=([0, 1], [0, 1])).T
    
    grad_Qhat = - Q_inv.T @ (grad_Q + grad_Q.T) @ Q
    grad_tensors[0] = grad_Qhat
    
    return tuple(grad_tensors)