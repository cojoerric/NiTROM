import os

import torch


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
