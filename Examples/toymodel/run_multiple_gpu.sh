#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=12355
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

torchrun --nproc_per_node=1 main.py