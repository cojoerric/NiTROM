import torch
import torch.distributed as dist
import os

def setup_distributed_gpus():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK'))

        dist.init_process_group(backend='nccl')

        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        return device, rank, world_size
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()