#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=v100:3
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH -o mmidas-logs/mmidas_%j.out
#SBATCH -e mmidas-logs/mmidas_%j.err
#SBATCH --time=96:00:00

source activate mdist-mmidas

python train.py --n_arm 7 --use-wandb --use_dist_sampler --n_epoch 500000
