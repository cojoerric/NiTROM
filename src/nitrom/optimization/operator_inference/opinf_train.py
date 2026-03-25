import torch
import torch.distributed as dist

from .opinf_model import OpInfModel


def train_opinf(
    model: OpInfModel,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    optimizer_type: str = "adam",
    print_every: int = 100,
    tol: float = 1e-10,
) -> OpInfModel:
    r"""
    Train an operator-inference model by minimizing the OpInf cost
    using analytic gradients.

    In a distributed setting (``torch.distributed`` initialized),
    gradients are averaged across ranks via ``all_reduce`` before
    each optimizer step.

    :param model: the OpInf model (already configured with training
        data, basis, regularization, etc.)
    :type model: OpInfModel
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
    :returns: the trained model
    :rtype: OpInfModel
    """
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    # Build optimizer
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(
            f"optimizer_type must be 'adam', 'sgd', or 'lbfgs', "
            f"got '{optimizer_type}'"
        )

    def _compute_and_assign_grads() -> torch.Tensor:
        """Evaluate cost, compute analytic gradients, and assign them."""
        optimizer.zero_grad()
        loss = model()
        grads = model.gradient()
        for param, grad in zip(model.parameters(), grads):
            param.grad = grad.clone()
        if is_distributed:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        return loss

    loss_prev = None

    for epoch in range(n_epochs):
        if optimizer_type == "lbfgs":
            loss = optimizer.step(_compute_and_assign_grads)
        else:
            loss = _compute_and_assign_grads()
            optimizer.step()

        loss_val = loss.item()
        if rank == 0 and (epoch % print_every == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch:6d} | Loss: {loss_val:.6e}")

        # Convergence check
        if loss_prev is not None and abs(loss_prev) > 0:
            rel_change = abs(loss_val - loss_prev) / abs(loss_prev)
            if rel_change < tol:
                if rank == 0:
                    print(
                        f"Converged at epoch {epoch} "
                        f"(rel. change {rel_change:.2e} < {tol:.2e}) "
                        f"loss value: {loss_val:.6e}"
                    )
                break
        loss_prev = loss_val

    return model
