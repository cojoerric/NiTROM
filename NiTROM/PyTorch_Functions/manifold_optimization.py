import torch


def project_manifold_grads(phi, psi, grad_phi, grad_psi):
    gphi = grad_phi - phi @ (phi.T @ grad_phi)
    sym = 0.5 * (psi.T @ grad_psi + grad_psi.T @ psi)
    gpsi = grad_psi - psi @ sym
    return gphi, gpsi


def qf(x):
    q, r = torch.linalg.qr(x, mode="reduced")
    diag = torch.sign(torch.diagonal(r))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    return q * diag


def project_tangent(point, vec, kind):
    if kind == "phi":
        return vec - point @ (point.T @ vec)
    sym = 0.5 * (point.T @ vec + vec.T @ point)
    return vec - point @ sym


def transport_optimizer_state(param, kind, optimizer, *, enabled=False, state_keys=()):
    if not enabled:
        return

    state = optimizer.state.get(param, None)
    if not state:
        return

    with torch.no_grad():
        for key in state_keys:
            buf = state.get(key, None)
            if (
                buf is not None
                and buf.shape == param.shape
                and buf.is_floating_point()
            ):
                buf.copy_(project_tangent(param, buf, kind))


def apply_qr_update(
    phi,
    psi,
    gphi,
    gpsi,
    lr,
    optimizer,
    *,
    transport_state=False,
    state_keys=(),
):
    r = phi.shape[1]
    phi_old = phi.clone()
    psi_old = psi.clone()
    phi.copy_(qf(phi - lr * gphi))
    psi.copy_(qf(psi - lr * gpsi))

    param_change = (
        torch.norm(phi - phi_old) / (r**0.5)
        + torch.norm(psi - psi_old) / (r**0.5)
    )
    if param_change > 10.0:
        phi.copy_(phi_old)
        psi.copy_(psi_old)
        return False

    transport_optimizer_state(
        phi,
        "phi",
        optimizer,
        enabled=transport_state,
        state_keys=state_keys,
    )
    transport_optimizer_state(
        psi,
        "psi",
        optimizer,
        enabled=transport_state,
        state_keys=state_keys,
    )
    return True
