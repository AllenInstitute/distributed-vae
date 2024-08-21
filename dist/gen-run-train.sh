#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <arms> <gpu> <epochs>"
    exit 1
fi

ARMS=$1
GPU=$2
EPOCHS=$3
FILE="train-scripts/run-train-A$ARMS-E$EPOCHS-$GPU.sh"



cat << EOF > $FILE
#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=$GPU:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH -o mmidas-logs/mmidas_%j.out
#SBATCH -e mmidas-logs/mmidas_%j.err
#SBATCH --time=24:00:00

source activate mdist-mmidas

python train.py --n_arm $ARMS --use-wandb --gpus 1 --n_epoch $EPOCHS
EOF