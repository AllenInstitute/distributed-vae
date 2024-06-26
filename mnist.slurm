#!/bin/bash
#SBATCH --job-name=fsdp_mnist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=slurm-log/out_%j.log
#SBATCH --error=slurm-log/error_%j.log
#SBATCH --partition=celltypes

mkdir -p slurm-log

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

eval "$(conda shell.bash hook)"
conda activate dist-mmidas

echo "running fsdp_mnist.py..."
srun python fsdp_mnist.py --multinode --backend nccl --epochs 1 --timeout 180
