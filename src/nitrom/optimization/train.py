import torch
import torch.distributed as dist

from .modules.base import InferenceModule


def _project_tangent(X: torch.Tensor, G: torch.Tensor, manifold: str) -> torch.Tensor:
    """Project Euclidean gradient G onto the tangent space at X."""
    if manifold == "grassmann":
        return G - X @ (X.T @ G)
    elif manifold == "stiefel":
        XtG = X.T @ G
        sym_XtG = 0.5 * (XtG + XtG.T)
        return G - X @ sym_XtG
    return G


def _retract(X: torch.Tensor) -> torch.Tensor:
    """Retract matrix X to the Stiefel/Grassmann manifold using QR decomposition."""
    Q, R = torch.linalg.qr(X)
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    ph = d.sign()
    ph[ph == 0] = 1.0
    return Q * ph.unsqueeze(-2)


def _transport_optimizer_states(optimizer, model, manifold_types) -> None:
    """Transport momentum buffers in optimizer state to the new tangent spaces."""
    for name, param in model.named_parameters():
        if name in manifold_types:
            state = optimizer.state.get(param)
            if state is None:
                continue
            for key in ["exp_avg", "momentum_buffer"]:
                if key in state:
                    buf = state[key]
                    proj_buf = _project_tangent(param, buf, manifold_types[name])
                    buf.copy_(proj_buf)


def train(
    model: InferenceModule,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    optimizer_type: str = "adam",
    print_every: int = 100,
    tol: float = 1e-10,
    n_restarts: int = 0,
    manifold_types: dict[str, str] | None = None,
) -> InferenceModule:
    r"""
    Train an :class:`InferenceModule` by minimizing its cost using the
    module's analytic gradients.

    Works for any concrete inference module (operator inference,
    polynomial-manifold inference, NiTROM, ...) because it relies only on
    the :class:`InferenceModule` contract: ``model()`` returns the scalar
    cost and ``model.gradient()`` returns the analytic gradients in
    :meth:`~torch.nn.Module.parameters` order.

    In a distributed setting (``torch.distributed`` initialized),
    gradients are averaged across ranks via ``all_reduce`` before
    each optimizer step.

    :param model: the inference module (already configured with training
        data, basis, regularization, etc.)
    :type model: InferenceModule
    :param n_epochs: number of optimization iterations
    :type n_epochs: int
    :param lr: learning rate
    :type lr: float
    :param optimizer_type: ``"adam"``, ``"sgd"``, or ``"lbfgs"``
    :type optimizer_type: str
    :param print_every: print the loss every *print_every* epochs
        (only on rank 0)
    :type print_every: int
    :param tol: convergence tolerance on the relative change in loss;
        training stops early if ``|loss - loss_prev| / |loss_prev| < tol``
    :type tol: float
    :param n_restarts: number of times to restart the optimizer (clearing its
        state, e.g. the L-BFGS history) when the relative-change criterion
        trips.  Quasi-Newton methods often *stall* at a non-stationary point --
        a line-search failure with a still-nonzero gradient -- and a fresh
        optimizer escapes it.  After a restart the run continues; it stops once
        a restart no longer reduces the loss or the restart budget is spent.
        ``0`` (default) reproduces the plain stop-on-stall behavior.
    :type n_restarts: int
    :param manifold_types: dict mapping parameter names to their manifold type,
        e.g., ``{"Phi": "grassmann", "Psi": "stiefel"}``. If ``None`` or empty,
        parameters are determined by the type of inference module.
    :type manifold_types: dict[str, str] or None
    :returns: the trained model
    :rtype: InferenceModule
    """
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    # Validate and normalize manifold types
    valid_manifolds = {"grassmann", "stiefel", "euclidean"}
    model_param_names = {name for name, _ in model.named_parameters()}

    # Automatically configure Phi to Grassmann and Psi to Stiefel for NitromModule
    default_manifold_types = {}
    if type(model).__name__ == "NitromModule":
        if "Phi" in model_param_names:
            default_manifold_types["Phi"] = "grassmann"
        if "Psi" in model_param_names:
            default_manifold_types["Psi"] = "stiefel"

    # Merge user settings, prioritizing user inputs
    user_manifold_types = manifold_types if manifold_types is not None else {}
    combined_manifold_types = {**default_manifold_types, **user_manifold_types}

    normalized_manifold_types = {}
    for name, mtype in combined_manifold_types.items():
        if name not in model_param_names:
            raise ValueError(
                f"Parameter '{name}' specified in manifold_types does not exist in the model."
            )
        mtype_lower = mtype.lower()
        if mtype_lower not in valid_manifolds:
            raise ValueError(
                f"Manifold type for parameter '{name}' must be one of {valid_manifolds}, "
                f"got '{mtype}'"
            )
        if mtype_lower != "euclidean":
            normalized_manifold_types[name] = mtype_lower

    # Synchronize parameters across ranks so every process starts the
    # data-parallel run from identical weights.  The all-reduced gradient is
    # only meaningful when all ranks sit at the same parameter point, which
    # would otherwise be violated by any per-rank randomness at construction.
    if is_distributed:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    # Build the optimizer (a fresh one is also created on each restart).
    def _make_optimizer():
        if optimizer_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        if optimizer_type == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        if optimizer_type == "lbfgs":
            return torch.optim.LBFGS(
                model.parameters(),
                lr=lr,
                max_iter=10,
                line_search_fn="strong_wolfe",
            )
        raise ValueError(
            f"optimizer_type must be 'adam', 'sgd', or 'lbfgs', "
            f"got '{optimizer_type}'"
        )

    optimizer = _make_optimizer()

    def _compute_and_assign_grads() -> torch.Tensor:
        """Evaluate cost, compute analytic gradients, and assign them."""
        # Retract manifold parameters to the manifold before evaluating cost
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in normalized_manifold_types:
                    param.copy_(_retract(param))

        optimizer.zero_grad()
        loss = model()
        grads = model.gradient()
        for param, grad in zip(model.parameters(), grads, strict=True):
            param.grad = grad.contiguous().clone()
        if is_distributed:
            # The cost and gradient are sums over trajectories whose weights
            # already carry the global normalization, so each rank holds a
            # partial sum and ReduceOp.SUM reconstructs the true global value
            # (no division by world_size).
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Reduce the loss too so the optimizer's line search and the
            # convergence test below operate on the same global cost on every
            # rank -- otherwise ranks could stop at different epochs and
            # deadlock on the next collective.
            loss = loss.detach().clone()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        # Project Euclidean gradients onto tangent spaces
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in normalized_manifold_types and param.grad is not None:
                    projected = _project_tangent(param, param.grad, normalized_manifold_types[name])
                    param.grad.copy_(projected)

        return loss

    loss_prev = None
    loss_at_last_restart = float("inf")
    restarts_left = n_restarts

    for epoch in range(n_epochs):
        if optimizer_type == "lbfgs":
            loss = optimizer.step(_compute_and_assign_grads)
        else:
            loss = _compute_and_assign_grads()
            optimizer.step()

        # Retract manifold parameters to the manifold after the step
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in normalized_manifold_types:
                    param.copy_(_retract(param))
                    if is_distributed:
                        dist.broadcast(param.data, src=0)

        # Transport optimizer state buffers (exp_avg, momentum_buffer)
        _transport_optimizer_states(optimizer, model, normalized_manifold_types)

        loss_val = loss.item()
        # Monitor: gradient norm at the current iterate (assigned param.grads).
        grad_norm = torch.sqrt(
            sum(
                (p.grad**2).sum()
                for p in model.parameters()
                if p.grad is not None
            )
        ).item()
        if rank == 0 and (epoch % print_every == 0 or epoch == n_epochs - 1):
            print(
                f"Epoch {epoch:6d} | Loss: {loss_val:.6e} "
                f"| GradNorm: {grad_norm:.6e}"
            )

        # Convergence check
        if loss_prev is not None and abs(loss_prev) > 0:
            rel_change = abs(loss_val - loss_prev) / abs(loss_prev)
            if rel_change < tol:
                # The optimizer stalled.  Restart it (clearing its state) if we
                # still have a restart budget and the last restart reduced the
                # loss; otherwise stop.
                improved = (
                    loss_at_last_restart == float("inf")
                    or loss_at_last_restart - loss_val
                    > tol * abs(loss_at_last_restart)
                )
                if restarts_left > 0 and improved:
                    restarts_left -= 1
                    loss_at_last_restart = loss_val
                    optimizer = _make_optimizer()
                    loss_prev = None
                    if rank == 0:
                        print(
                            f"Restart {n_restarts - restarts_left}/{n_restarts} "
                            f"at epoch {epoch} (loss {loss_val:.6e})"
                        )
                    continue
                if rank == 0:
                    print(
                        f"Converged at epoch {epoch} "
                        f"(rel. change {rel_change:.2e} < {tol:.2e}) "
                        f"loss value: {loss_val:.6e}"
                    )
                break
        loss_prev = loss_val

    return model
