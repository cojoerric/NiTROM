#!/bin/bash

module restore gpu
source /scratch/projects/compilers/intel24.0/oneapi/intelpython/python3.9/etc/profile.d/conda.sh
conda activate nitrom

export MASTER_PORT=12355
export MASTER_ADDR=$(hostname)
export IBRUN_TASKS_PER_NODE=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=hfi1_0
export NCCL_SOCKET_IFNAME=ibp92s0

ibrun -n $SLURM_NNODES torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  gradient_profiling.py