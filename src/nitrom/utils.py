import torch
import os

def interp_quadratic(
    t_eval: torch.Tensor,
    t_data: torch.Tensor,
    y_data: torch.Tensor,
) -> torch.Tensor:
    r"""
    Piecewise quadratic (3-point Lagrange) interpolation of uniformly
    sampled data.

    For each query point in *t_eval*, the three nearest data points are
    used to build a degree-2 Lagrange polynomial:

    .. math::

        p(t) = \sum_{j=0}^{2} y_j \prod_{\substack{m=0 \\ m \neq j}}^{2}
               \frac{t - t_m}{t_j - t_m}

    The data in *y_data* may have arbitrary leading dimensions (e.g.
    ``(n, n_data)`` or ``(B, n, n_data)``); interpolation is always
    performed along the **last** axis.

    :param t_eval: query times of shape ``(n_eval,)``
    :type t_eval: torch.Tensor
    :param t_data: data times of shape ``(n_data,)``, must be sorted
    :type t_data: torch.Tensor
    :param y_data: data values with time along the last axis,
        shape ``(..., n_data)``
    :type y_data: torch.Tensor
    :returns: interpolated values, shape ``(..., n_eval)``
    :rtype: torch.Tensor
    """
    n_data = t_data.shape[0]

    # Find the index of the right neighbour for each query point
    idx = torch.searchsorted(t_data, t_eval).clamp(1, n_data - 1)

    # Choose the centre index of the 3-point stencil, clamped so that
    # i-1, i, i+1 are all valid
    ic = idx.clamp(1, n_data - 2)

    t0 = t_data[ic - 1]  # (n_eval,)
    t1 = t_data[ic]
    t2 = t_data[ic + 1]

    # Lagrange basis values at t_eval
    L0 = ((t_eval - t1) * (t_eval - t2)) / ((t0 - t1) * (t0 - t2))
    L1 = ((t_eval - t0) * (t_eval - t2)) / ((t1 - t0) * (t1 - t2))
    L2 = ((t_eval - t0) * (t_eval - t1)) / ((t2 - t0) * (t2 - t1))

    # Gather data values at the stencil points: (..., n_eval)
    y0 = y_data[..., ic - 1]
    y1 = y_data[..., ic]
    y2 = y_data[..., ic + 1]

    return y0 * L0 + y1 * L1 + y2 * L2

def setup_distributed() -> tuple[torch.device, int, int]:
    r"""
    Initialize the PyTorch distributed process group for multi-GPU or
    multi-CPU training.

    When launched via ``torchrun``, the environment variables ``RANK``,
    ``LOCAL_RANK``, and ``WORLD_SIZE`` are read automatically.  The backend
    is selected based on hardware: **nccl** when CUDA is available,
    **gloo** otherwise.

    In single-process mode (no ``torchrun``), falls back to a local
    CUDA or CPU device with rank 0 and world size 1.

    :returns: ``(device, rank, world_size)``
    :rtype: tuple[torch.device, int, int]
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        init_kwargs = {
            "backend": backend,
            "init_method": "env://",
            "rank": rank,
            "world_size": world_size,
        }
        if backend == "nccl":
            init_kwargs["device_id"] = device

        torch.distributed.init_process_group(**init_kwargs)

        if backend == "nccl":
            torch.distributed.barrier(device_ids=[local_rank])
        else:
            torch.distributed.barrier()

        return device, rank, world_size
    else:
        # single-process fallback
        if torch.cuda.is_available():
            return torch.device("cuda"), 0, 1
        else:
            return torch.device("cpu"), 0, 1


def cleanup_distributed() -> None:
    """
    Destroy the distributed process group if one is active.

    Should be called at the end of the training script to release
    distributed resources cleanly.
    """
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()