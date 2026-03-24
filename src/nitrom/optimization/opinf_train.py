import torch
import torch.distributed as dist

from .opinf_cost_and_grad import OpInfCostAndGrad
from .opinf_nn_models import OpInfParams, GasOpInfParams, OpInfForwardModule


def train_opinf(
    opinf_class: OpInfCostAndGrad,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    optimizer_type: str = "adam",
    print_every: int = 100,
    tol: float = 1e-10,
) -> OpInfParams | GasOpInfParams:
    r"""
    Train an operator-inference model by minimizing the OpInf cost
    using analytic gradients.

    The training loop:

    1. Builds an :class:`OpInfParams` (or :class:`GasOpInfParams` if
       ``gas_flag`` is set) module from the initial guess.
    2. Wraps it in an :class:`OpInfForwardModule` for cost/gradient
       evaluation.
    3. Runs gradient descent for *n_epochs* iterations, using the
       analytic gradient from :meth:`OpInfCostAndGrad.gradient` to
       update the parameters.

    In a distributed setting (``torch.distributed`` initialized),
    gradients are averaged across ranks via ``all_reduce`` before
    each optimizer step.

    :param opinf_class: the OpInf cost-and-gradient object (already
        configured with training data, basis, regularization, etc.)
    :type opinf_class: OpInfCostAndGrad
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
    :returns: the trained parameter module
    :rtype: OpInfParams | GasOpInfParams
    """
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    # Build parameter module
    if opinf_class.gas_flag:
        params_module = GasOpInfParams(opinf_class)
    else:
        params_module = OpInfParams(opinf_class)

    # Build forward module
    forward_module = OpInfForwardModule(opinf_class, params_module)

    # Build optimizer
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(params_module.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(params_module.parameters(), lr=lr)
    elif optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params_module.parameters(),
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
        loss = forward_module.forward()
        grads = forward_module.gradient()
        for param, grad in zip(params_module.parameters(), grads):
            param.grad = grad.clone()
        if is_distributed:
            for param in params_module.parameters():
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

    return params_module
