import torch


def create_intitial_guess(A, H=None, r=None):
    """
    Build initialization tensors for the globally stable model from A (and optional H).
    The construction uses Q = I, J = skew(A), R = -sym(A) projected to PSD, and Hhat = 0.5 * H.
    Note: the resulting H corresponds to the antisymmetric part of the input H.
    """
    if r is None:
        r = A.shape[0]

    device = A.device
    dtype = A.dtype

    # Initialize Q as identity
    Qhat = torch.eye(r, dtype=dtype, device=device)

    # Decompose A into symmetric and skew-symmetric parts
    A_sym = 0.5 * (A + A.T)
    A_skew = 0.5 * (A - A.T)

    # Make sure symmetric part is negative definite for stability
    eigvals, eigvecs = torch.linalg.eigh(A_sym)
    neg_eigvals = torch.where(
        eigvals < 0, eigvals, -0.1*torch.ones_like(eigvals))
    A_sym_stable = eigvecs @ torch.diag(neg_eigvals) @ eigvecs.T

    # J = Jhat - Jhat.T -> choose Jhat so J matches the skew part of A
    Jhat = 0.5 * A_skew

    R_mat = -A_sym_stable
    try:
        Rhat = torch.linalg.cholesky(R_mat)
    except:
        # If Cholesky fails, use eigenvector decomposition
        eigvals, eigvecs = torch.linalg.eigh(R_mat)
        Rhat = eigvecs @ torch.diag(torch.sqrt(
            torch.clamp(eigvals, min=1e-6))) @ eigvecs.T

    # For H tensor, either use the provided H or initialize to zeros
    if H is not None:
        Hhat = 0.5 * H.clone()
    else:
        Hhat = torch.zeros((r, r, r), dtype=dtype, device=device)

    return {
        'Jhat': Jhat,
        'Rhat': Rhat,
        'Qhat': Qhat,
        'Hhat': Hhat
    }

def compute_Q(Qhat):
    Q_inv = torch.linalg.inv(Qhat)
    Q = Q_inv @ Q_inv.T
    return Q, Q_inv

def compute_JR(Jhat, Rhat):
    J = Jhat - Jhat.T
    R = Rhat @ Rhat.T
    return J, R

def compute_H(Hhat, Q):
    H2 = Hhat.permute(2, 1, 0)
    M_tensor = Hhat - H2
    H = M_tensor @ Q
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
        J, R = compute_JR(Jhat, Rhat)
        A = (J - R) @ Q
        tensors_new.append(A)
        other_tensors.extend([J, R])
    if 2 in poly_comp:
        Hhat = tensors[-1]
        H = compute_H(Hhat, Q)
        tensors_new.append(H)
    
    return tuple(tensors_new), tuple(other_tensors)

def propagate_gradients(grads, tensors, tensors_hat, poly_comp):
    grad_tensors = [torch.zeros_like(tensor_hat) for tensor_hat in tensors_hat]
    Q = tensors[0]; Q_inv = tensors[1]
    grad_Q = torch.zeros_like(Q)

    if 1 in poly_comp:
        J = tensors[2]; R = tensors[3]
        Rhat = tensors_hat[2]
        grad_A = grads[0]
        grad_J = grad_A @ Q.T; grad_R = -grad_J; grad_Q += (J - R).T @ grad_A
        grad_Jhat = grad_J - grad_J.T; grad_Rhat = (grad_R + grad_R.T) @ Rhat
        grad_tensors[1] = grad_Jhat; grad_tensors[2] = grad_Rhat
    if 2 in poly_comp:
        Hhat = tensors_hat[-1]
        grad_H = grads[-1]
        S = torch.tensordot(grad_H, Q, dims=([2],[0]))
        grad_Hhat = S - S.permute(2, 1, 0)
        grad_tensors[-1] = grad_Hhat

        G = Hhat - Hhat.permute(2, 1, 0)
        grad_Q += torch.tensordot(grad_H, G, dims=([0, 1], [0, 1])).T
    
    grad_Qhat = - Q_inv.T @ (grad_Q + grad_Q.T) @ Q
    grad_tensors[0] = grad_Qhat
    
    return tuple(grad_tensors)

def finite_difference_gradcheck(model, n_samples=10, eps=1e-6, seed=0):
    torch.manual_seed(seed)
    cost_fn = model.cost_fn
    grad_fn = model.grad_fn
    params = model.params
    grads = grad_fn(*model.param_tuple())

    grad_map = dict(zip(["Phi", "Psi"] + params.tensor_names(), grads))
    tensor_names = params.tensor_names()
    print("Finite-difference check (tensors only):")

    with torch.no_grad():
        for name in tensor_names:
            tensor = getattr(params, name)
            grad = grad_map[name]
            if not isinstance(grad, torch.Tensor):
                grad = torch.tensor(grad, device=tensor.device, dtype=tensor.dtype)

            mean_abs_err = 0.0
            mean_rel_err = 0.0
            for _ in range(n_samples):
                idx = tuple(torch.randint(0, s, (1,), device=tensor.device).item() for s in tensor.shape)
                orig = tensor[idx].item()

                tensor[idx] = orig + eps
                f_plus = cost_fn(*model.param_tuple()).item()
                tensor[idx] = orig - eps
                f_minus = cost_fn(*model.param_tuple()).item()
                tensor[idx] = orig

                fd = (f_plus - f_minus) / (2.0 * eps)
                ad = grad[idx].item()
                err = abs(fd - ad)
                mean_abs_err += err
                rel_err = err / (abs(ad) + 1e-12)
                mean_rel_err += rel_err

            mean_abs_err /= n_samples
            mean_rel_err /= n_samples
            print(f"  {name}: mean_abs_err={mean_abs_err:.3e}, mean_rel_err={mean_rel_err:.3e}")
