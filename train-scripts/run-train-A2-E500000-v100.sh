#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=v100:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH -o mmidas-logs/mmidas_%j.out
#SBATCH -e mmidas-logs/mmidas_%j.err
#SBATCH --time=96:00:00

source ~/.bashrc

mamba activate mdist-mmidas

python train.py --n_arm 2 --use-wandb --gpus 1 --n_epoch 500000
