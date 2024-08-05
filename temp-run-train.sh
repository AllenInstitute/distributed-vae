#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=a100:1
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH -o mmidas-logs/mmidas_%j.out
#SBATCH -e mmmidas-logs/mmmidas_%j.err

source /home/hilal.mufti/miniforge3/etc/profile.d/conda.sh
/home/hilal.mufti/miniforge3/bin/mamba activate mdist-mmidas

python train.py --n_arms 1 --n_epoch 3
