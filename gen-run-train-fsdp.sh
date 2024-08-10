#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <arms> <gpu> <world-size>"
    exit 1
fi

ARMS=$1
GPU=$2
WORLD_SIZE=$3
FILE="train-scripts/run-train-fsdp-A$ARMS-$GPU.sh"

cat << EOF > $FILE
#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=$GPU:$WORLD_SIZE
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH -o mmidas-logs/mmidas_%j.out
#SBATCH -e mmidas-logs/mmidas_%j.err
#SBATCH --time=24:00:00

source activate mdist-mmidas

python train.py --n_arm $ARMS --use-wandb --use_dist_sampler
EOF