import torch
import torch.distributed as dist
import os


def setup_distributed():
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

        dist.init_process_group(**init_kwargs)

        if backend == "nccl":
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

        return device, rank, world_size

    # single-process fallback
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, 1
    else:
        return torch.device("cpu"), 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
