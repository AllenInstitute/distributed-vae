#!/bin/bash

#SBATCH --job-name=fsdp-mnist
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00
#SBATCH --output=slurm-log/out_%j.log
#SBATCH --error=slurm-log/error_%j.log
#SBATCH --partition=celltypes

# srun --nodes=1 --gres=gpu:2 nvidia-smi

echo "current nodes: $SLURM_NODELIST"
mkdir -p slurm-log

export MASTER_PORT=12355
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE=$WORLD_SIZE"

master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "MASTER_ADDR=$master_addr"

eval "$(conda shell.bash hook)"
conda activate dist-mmidas

srun python fsdp.py --task mnist --multinode


# torchrun --nproc_per_node=2 --nnodes=2 --node_rank=$SLURM_NODEID --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) --master_port=12355 fsdp.py --task trivial --multinode


# DONT DELETE
# srun -G6 -p celltypes --constraint="v100|titanxp|1080ti|titanx" bash -c 'hostname && nvidia-smi -L'