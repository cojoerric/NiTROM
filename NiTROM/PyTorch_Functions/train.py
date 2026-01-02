import torch
import torch.distributed as dist

def train_model(
    model,
    pool,
    optimizer,
    num_epochs,
    *,
    scheduler=None,
    log_every=10,
    grad_clip=1e3,
    manifold_retraction="qr",
    manifold_lr=None,
    callback=None,
):
    if hasattr(model, "cost_fn") and hasattr(model, "grad_fn"):
        cost_fn, grad_fn = model.cost_fn, model.grad_fn
    else:
        raise ValueError("Model must have cost_fn and grad_fn attributes.")

    for p in model.parameters():
        p.requires_grad_(False)

    is_dist = (
        getattr(pool, "world_size", 1) > 1
        and dist.is_available()
        and dist.is_initialized()
    )
    is_lbfgs = optimizer.__class__.__name__ == "LBFGS"

    history = []
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)

        last_grad_norm = 0.0
        last_loss = None
        last_gphi = None
        last_gpsi = None

        def _project_manifold_grads(phi, psi, grad_phi, grad_psi):
            gphi = grad_phi
            if not isinstance(gphi, torch.Tensor):
                gphi = torch.tensor(gphi, device=phi.device, dtype=phi.dtype)
            gpsi = grad_psi
            if not isinstance(gpsi, torch.Tensor):
                gpsi = torch.tensor(gpsi, device=psi.device, dtype=psi.dtype)

            gphi = gphi - phi @ (phi.T @ gphi)
            sym = 0.5 * (psi.T @ gpsi + gpsi.T @ psi)
            gpsi = gpsi - psi @ sym
            return gphi, gpsi

        def _qf(x):
            q, r = torch.linalg.qr(x, mode="reduced")
            diag = torch.sign(torch.diagonal(r))
            diag = torch.where(diag == 0, torch.ones_like(diag), diag)
            return q * diag

        def _apply_qr_update(phi, psi, gphi, gpsi, lr):
            phi.copy_(_qf(phi - lr * gphi))
            psi.copy_(_qf(psi - lr * gpsi))

        def _compute_and_set_grads():
            nonlocal last_grad_norm, last_loss, last_gphi, last_gpsi
            cost_val = cost_fn(*model.param_tuple())
            grads = grad_fn(*model.param_tuple())

            if is_dist:
                cost_val = cost_val.contiguous()
                dist.all_reduce(cost_val, op=dist.ReduceOp.SUM)
                grads = tuple(g.contiguous() for g in grads)
                for g in grads:
                    dist.all_reduce(g, op=dist.ReduceOp.SUM)

            for p, g in zip(params.tensors(), grad_tensors):
                p.grad = g if isinstance(g, torch.Tensor) else torch.tensor(g, device=p.device, dtype=p.dtype)

            params = model.params
            if model.name == 'nitrom':
                grad_Phi = grads[0]
                grad_Psi = grads[1]
                grad_tensors = grads[2:]

                if manifold_retraction == "qr":
                    for p in (params.Phi, params.Psi):
                        p.grad = None
                else:
                    params.Phi.grad = (
                        grad_Phi if isinstance(grad_Phi, torch.Tensor)
                        else torch.tensor(grad_Phi, device=params.Phi.device, dtype=params.Phi.dtype)
                    )
                    params.Psi.grad = (
                        grad_Psi if isinstance(grad_Psi, torch.Tensor)
                        else torch.tensor(grad_Psi, device=params.Psi.device, dtype=params.Psi.dtype)
                    )
            
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += (p.grad.data ** 2).mean().item()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            if model.name == 'nitrom':
                if manifold_retraction == "qr":
                    phi = params.Phi
                    psi = params.Psi
                    gphi, gpsi = _project_manifold_grads(phi, psi, grad_Phi, grad_Psi)
                    last_gphi, last_gpsi = gphi, gpsi

                    grad_norm += (gphi.data ** 2).mean().item() + (gpsi.data ** 2).mean().item()
                    if not is_lbfgs:
                        lr = manifold_lr
                        if lr is None:
                            lr = optimizer.param_groups[0]["lr"]
                        with torch.no_grad():
                            _apply_qr_update(phi, psi, gphi, gpsi, lr)

            last_grad_norm = grad_norm ** 0.5
            last_loss = cost_val
            return cost_val

        if is_lbfgs:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                return _compute_and_set_grads()

            optimizer.step(closure)
            if manifold_retraction == "qr":
                lr = manifold_lr
                if lr is None:
                    lr = optimizer.param_groups[0]["lr"]
                with torch.no_grad():
                    if model.name == 'nitrom':
                        if last_gphi is not None and last_gpsi is not None:
                            _apply_qr_update(model.params.Phi, model.params.Psi, last_gphi, last_gpsi, lr)
            cost_val = last_loss
            grad_norm = last_grad_norm
        else:
            cost_val = _compute_and_set_grads()
            grad_norm = last_grad_norm

            optimizer.step()

        loss = float(cost_val.detach().item())
        history.append(loss)

        if scheduler:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(loss)
            else:
                scheduler.step()

        if log_every and pool.rank == 0:
            if epoch % log_every == 0 or epoch == num_epochs - 1:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.6e}, LR: {lr:.3e}, Grad Norm: {grad_norm:.3e}"
                )

        if callback is not None:
            callback(epoch, loss, model)

    return model, history
