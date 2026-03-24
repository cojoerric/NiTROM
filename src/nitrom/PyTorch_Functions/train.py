import torch
import torch.distributed as dist
import copy
from .line_search import LineSearch


def train_model(
    model,
    pool,
    optimizer,
    num_epochs,
    *,
    scheduler=None,
    log_every=10,
    grad_clip=0.1,
    manifold_retraction="qr",
    manifold_lr=None,
    callback=None,
    lbfgs_updates_manifold=False,
    use_line_search=False,
    line_search_params=None,
    safe_step=False,
    ss_max_retries=5,
    ss_shrink=0.5,
    ss_tol=0.0,
    ss_keep_lr=True,
    vector_transport=False,
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
    is_nitrom = model.name == "nitrom"
    if use_line_search:
        _ls_params = line_search_params or {}
        _ls_user_set_alpha_init = "alpha_init" in _ls_params
        line_searcher = LineSearch(**_ls_params)

    _vt_state_keys = (
        "momentum_buffer",
        "exp_avg",
        "exp_avg_hat",
        "search_direction",
        "velocity",
        "dir_buffer",
    )

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

        def _project_tangent(point, vec, kind):
            if kind == "phi":
                return vec - point @ (point.T @ vec)
            sym = 0.5 * (point.T @ vec + vec.T @ point)
            return vec - point @ sym

        def _transport_optimizer_state(param, kind):
            if (not vector_transport) or (not is_nitrom) or manifold_retraction != "qr":
                return
            state = optimizer.state.get(param, None)
            if not state:
                return
            with torch.no_grad():
                for key in _vt_state_keys:
                    buf = state.get(key, None)
                    if (
                        isinstance(buf, torch.Tensor)
                        and buf.shape == param.shape
                        and buf.is_floating_point()
                    ):
                        buf.copy_(_project_tangent(param, buf, kind))

        def _apply_qr_update(phi, psi, gphi, gpsi, lr):
            r = phi.shape[1]
            phi_old = phi.clone()
            psi_old = psi.clone()
            phi.copy_(_qf(phi - lr * gphi))
            psi.copy_(_qf(psi - lr * gpsi))

            param_change = torch.norm(phi - phi_old) / (r**0.5) + torch.norm(
                psi - psi_old
            ) / (r**0.5)
            if param_change > 10.0:
                phi.copy_(phi_old)
                psi.copy_(psi_old)
                return False

            _transport_optimizer_state(phi, "phi")
            _transport_optimizer_state(psi, "psi")
            return True

        def _compute_and_set_grads():
            nonlocal last_grad_norm, last_loss, last_gphi, last_gpsi
            cost_val = cost_fn(*model.param_tuple())
            if not isinstance(cost_val, torch.Tensor):
                ref = next(model.parameters())
                cost_val = torch.tensor(cost_val, device=ref.device, dtype=ref.dtype)
            grads = grad_fn(*model.param_tuple())

            if is_dist:
                cost_val = cost_val.contiguous()
                dist.all_reduce(cost_val, op=dist.ReduceOp.SUM)
                grads = tuple(g.contiguous() for g in grads)
                for g in grads:
                    dist.all_reduce(g, op=dist.ReduceOp.SUM)

            params = model.params
            n_tensors = len(params.tensor_names())
            if model.name == "nitrom":
                if len(grads) != n_tensors + 2:
                    raise ValueError(
                        "Unexpected gradient tuple size. Expected tensors + Phi + Psi."
                    )
                grad_Phi = grads[0]
                grad_Psi = grads[1]
                grad_tensors = grads[2:]
            else:
                grad_tensors = grads

            for p, g in zip(params.tensors(), grad_tensors):
                p.grad = (
                    g
                    if isinstance(g, torch.Tensor)
                    else torch.tensor(g, device=p.device, dtype=p.dtype)
                )

            if is_nitrom and manifold_retraction == "qr":
                gphi, gpsi = _project_manifold_grads(
                    params.Phi, params.Psi, grad_Phi, grad_Psi
                )
                last_gphi, last_gpsi = gphi, gpsi

                if is_lbfgs and lbfgs_updates_manifold:
                    params.Phi.grad = gphi
                    params.Psi.grad = gpsi
                else:
                    params.Phi.grad = None
                    params.Psi.grad = None
            elif is_nitrom:
                params.Phi.grad = (
                    grad_Phi
                    if isinstance(grad_Phi, torch.Tensor)
                    else torch.tensor(
                        grad_Phi, device=params.Phi.device, dtype=params.Phi.dtype
                    )
                )
                params.Psi.grad = (
                    grad_Psi
                    if isinstance(grad_Psi, torch.Tensor)
                    else torch.tensor(
                        grad_Psi, device=params.Psi.device, dtype=params.Psi.dtype
                    )
                )

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += (p.grad.data**2).mean().item()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            if is_nitrom and manifold_retraction == "qr":
                grad_norm += (gphi.data**2).mean().item() + (gpsi.data**2).mean().item()

            last_grad_norm = grad_norm**0.5
            last_loss = cost_val
            return cost_val

        if is_lbfgs:
            params = model.params

            def closure():
                optimizer.zero_grad(set_to_none=True)
                if is_nitrom and manifold_retraction == "qr" and lbfgs_updates_manifold:
                    with torch.no_grad():
                        params.Phi.copy_(_qf(params.Phi))
                        params.Psi.copy_(_qf(params.Psi))
                        _transport_optimizer_state(params.Phi, "phi")
                        _transport_optimizer_state(params.Psi, "psi")
                return _compute_and_set_grads()

            optimizer.step(closure)

            if is_nitrom and manifold_retraction == "qr":
                if lbfgs_updates_manifold:
                    with torch.no_grad():
                        params.Phi.copy_(_qf(params.Phi))
                        params.Psi.copy_(_qf(params.Psi))
                        _transport_optimizer_state(params.Phi, "phi")
                        _transport_optimizer_state(params.Psi, "psi")
                else:
                    if last_gphi is not None and last_gpsi is not None:
                        lr = manifold_lr
                        if lr is None:
                            lr = optimizer.param_groups[0]["lr"] * 0.1
                        with torch.no_grad():
                            _apply_qr_update(
                                model.params.Phi,
                                model.params.Psi,
                                last_gphi,
                                last_gpsi,
                                lr,
                            )
            cost_val = last_loss
            loss_for_log = float(cost_val.detach().item())
            grad_norm = last_grad_norm
            print_out = f"Epoch [{epoch + 1}/{num_epochs}], Loss: {float(cost_val.detach().item()):.4e}, Grad Norm: {grad_norm:.3e}"
        else:
            cost_val = _compute_and_set_grads()
            grad_norm = last_grad_norm
            lr_tensors_used = optimizer.param_groups[0]["lr"]
            lr_manifold_used = (
                manifold_lr if manifold_lr is not None else lr_tensors_used
            )

            if use_line_search and not is_lbfgs:
                params = model.params

                lr_tensors = optimizer.param_groups[0]["lr"]
                lr_manifold = (
                    manifold_lr
                    if manifold_lr is not None
                    else optimizer.param_groups[0]["lr"]
                )
                if (not _ls_user_set_alpha_init) and hasattr(
                    line_searcher, "min_alpha"
                ):
                    line_searcher.alpha_init = float(max(lr_tensors, lr_manifold))

                param_list = []
                direction_list = []

                if is_nitrom and manifold_retraction == "qr":
                    if last_gphi is None or last_gpsi is None:
                        raise RuntimeError(
                            "Expected manifold gradients to be computed for line search."
                        )
                    param_list.extend([params.Phi, params.Psi])
                    direction_list.extend(
                        [lr_manifold * last_gphi, lr_manifold * last_gpsi]
                    )

                for p in params.tensors():
                    if p.grad is None:
                        continue
                    param_list.append(p)
                    direction_list.append(lr_tensors * p.grad)

                grad_norm_sq = 0.0
                if is_nitrom and manifold_retraction == "qr":
                    grad_norm_sq += lr_manifold * (last_gphi.detach() ** 2).sum().item()
                    grad_norm_sq += lr_manifold * (last_gpsi.detach() ** 2).sum().item()
                for p in params.tensors():
                    if p.grad is not None:
                        grad_norm_sq += lr_tensors * (p.grad.detach() ** 2).sum().item()

                def cost_fn_trial(trial_params):
                    with torch.no_grad():
                        old_vals = [p.clone() for p in param_list]

                        k = 0
                        if is_nitrom and manifold_retraction == "qr":
                            params.Phi.copy_(_qf(trial_params[0]))
                            params.Psi.copy_(_qf(trial_params[1]))
                            k = 2

                        tensor_trials = trial_params[k:]
                        tcount = 0
                        for p in params.tensors():
                            if p.grad is None:
                                continue
                            p.copy_(tensor_trials[tcount])
                            tcount += 1

                        loss_val = cost_fn(*model.param_tuple())
                        if is_dist:
                            loss_val = loss_val.contiguous()
                            dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)

                        for p, v in zip(param_list, old_vals):
                            p.copy_(v)

                    return loss_val

                alpha, trial_loss, _ = line_searcher.search(
                    cost_fn_trial,
                    param_list,
                    direction_list,
                    cost_val,
                    grad_norm_sq,
                )
                lr_tensors_used = float(alpha) * float(lr_tensors)
                lr_manifold_used = float(alpha) * float(lr_manifold)

                with torch.no_grad():
                    accepted = [
                        p - alpha * d for p, d in zip(param_list, direction_list)
                    ]
                    if is_nitrom and manifold_retraction == "qr":
                        params.Phi.copy_(_qf(accepted[0]))
                        params.Psi.copy_(_qf(accepted[1]))
                        _transport_optimizer_state(params.Phi, "phi")
                        _transport_optimizer_state(params.Psi, "psi")
                        offset = 2
                    else:
                        offset = 0

                    tensor_trials = accepted[offset:]
                    tcount = 0
                    for p in params.tensors():
                        if p.grad is None:
                            continue
                        p.copy_(tensor_trials[tcount])
                        tcount += 1
                loss_for_log = float(trial_loss.detach().item())
            else:
                params = model.params

                def _eval_loss():
                    lv = cost_fn(*model.param_tuple())
                    if is_dist:
                        lv = lv.contiguous()
                        dist.all_reduce(lv, op=dist.ReduceOp.SUM)
                    return lv

                step_params = []
                if is_nitrom:
                    step_params.extend([params.Phi, params.Psi])
                step_params.extend(params.tensors())

                with torch.no_grad():
                    base_vals = [p.detach().clone() for p in step_params]

                def _restore(vals):
                    with torch.no_grad():
                        for p, v in zip(step_params, vals):
                            p.copy_(v)

                if safe_step:
                    base_opt_state = copy.deepcopy(optimizer.state_dict())
                    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
                    base_loss = float(cost_val.detach().item())

                    accepted_loss = None
                    accepted_lr = None
                    accepted_scale = None

                    tries = max(1, int(ss_max_retries))
                    for attempt in range(tries):
                        scale = float(ss_shrink) ** attempt

                        _restore(base_vals)
                        optimizer.load_state_dict(base_opt_state)
                        for pg, lr0 in zip(optimizer.param_groups, base_lrs):
                            pg["lr"] = lr0 * scale

                        if is_nitrom and manifold_retraction == "qr":
                            if last_gphi is None or last_gpsi is None:
                                raise RuntimeError(
                                    "Expected manifold gradients for safe step."
                                )
                            lr_m0 = (
                                manifold_lr if manifold_lr is not None else base_lrs[0]
                            )
                            lr_m = lr_m0 * scale
                            with torch.no_grad():
                                ok = _apply_qr_update(
                                    params.Phi, params.Psi, last_gphi, last_gpsi, lr_m
                                )
                            if not ok:
                                continue

                        optimizer.step()
                        trial_loss = _eval_loss()
                        trial_loss_val = float(trial_loss.detach().item())

                        if trial_loss_val <= base_loss + float(ss_tol):
                            accepted_loss = trial_loss
                            accepted_lr = optimizer.param_groups[0]["lr"]
                            accepted_scale = scale
                            break

                    if accepted_loss is None:
                        _restore(base_vals)
                        optimizer.load_state_dict(base_opt_state)
                        for pg, lr0 in zip(optimizer.param_groups, base_lrs):
                            pg["lr"] = lr0
                        loss_for_log = base_loss
                        lr_manifold_used = 0.0
                        lr_tensors_used = 0.0
                    else:
                        loss_for_log = float(accepted_loss.detach().item())
                        lr_tensors_used = float(accepted_lr)

                        if is_nitrom and manifold_retraction == "qr":
                            lr_m0 = (
                                manifold_lr if manifold_lr is not None else base_lrs[0]
                            )
                            lr_manifold_used = float(lr_m0) * float(accepted_scale)
                        else:
                            lr_manifold_used = (
                                manifold_lr
                                if manifold_lr is not None
                                else lr_tensors_used
                            )

                        if not ss_keep_lr:
                            for pg, lr0 in zip(optimizer.param_groups, base_lrs):
                                pg["lr"] = lr0
                else:
                    if is_nitrom and manifold_retraction == "qr":
                        if last_gphi is None or last_gpsi is None:
                            raise RuntimeError(
                                "Expected manifold gradients for QR update."
                            )
                        lr_m = (
                            manifold_lr
                            if manifold_lr is not None
                            else optimizer.param_groups[0]["lr"]
                        )
                        with torch.no_grad():
                            _apply_qr_update(
                                params.Phi, params.Psi, last_gphi, last_gpsi, lr_m
                            )

                    optimizer.step()
                    loss_for_log = float(_eval_loss().detach().item())
                    lr_tensors_used = float(optimizer.param_groups[0]["lr"])
                    lr_manifold_used = (
                        float(manifold_lr)
                        if manifold_lr is not None
                        else lr_tensors_used
                    )

            print_out = (
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_for_log:.4e}, "
                f"LR: {lr_tensors_used:.3e}, mLR: {lr_manifold_used:.3e}, "
                f"Grad Norm: {grad_norm:.3e}"
            )

        loss = loss_for_log
        history.append(loss)

        if scheduler:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(loss)
            else:
                scheduler.step()

        if log_every and pool.rank == 0:
            if epoch % log_every == 0 or epoch == num_epochs - 1:
                print(print_out)

        if callback is not None:
            callback(epoch, loss, model)

    return model, history
