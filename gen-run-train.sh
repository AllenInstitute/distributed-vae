#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <arms> <gpu>"
    exit 1
fi

ARMS=$1
GPU=$2
FILE="train-scripts/run-train-A$ARMS-$GPU.sh"



cat << EOF > $FILE
#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=$GPU:1
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH -o mmidas-logs/mmidas_%j.out
#SBATCH -e mmidas-logs/mmidas_%j.err
#SBATCH --time=24:00:00

source activate mdist-mmidas

python ../train.py --n_arm $ARMS --use-wandb
EOF