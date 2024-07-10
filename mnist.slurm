#!/bin/bash

#SBATCH --job-name=fsdp-mnist
#SBATCH --partition=celltypes
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-log/out_%j.log
#SBATCH --error=slurm-log/error_%j.log

export MASTER_PORT=12340
export WORLD_SIZE=4

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/miniforge3/etc/profile.d/mamba.sh
mamba activate mdist-mmidas

srun python fsdp_mnist_srun.py