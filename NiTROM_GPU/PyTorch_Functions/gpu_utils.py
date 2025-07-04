import torch
import torch.distributed as dist
import os

def setup_distributed_gpus():
    if torch.cuda.device_count() > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        return device, dist.get_rank(), dist.get_world_size()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()