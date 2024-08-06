#!/bin/bash

if [ "$#" -ne 11 ]; then
    echo "Usage: $0 <epochs> <batch_size> <model> <gpus> <use_dist_sampler> <use_batchnorm> <use_orig_params> <use_sync> <sharding_strategy> <use_compile> <mixed-precision>"
    exit 1
fi

EPOCHS=$1
BATCH_SIZE=$2
MODEL=$3
GPUS=$4
USE_DIST_SAMPLER=$5
USE_BATCHNORM=$6
USE_ORIG_PARAMS=$7
USE_SYNC=$8
STRAT=$9
USE_COMPILE=${10}
MIXED=${11}


# Prepare the use_dist_sampler argument
if [ "$USE_DIST_SAMPLER" = "True" ]; then
    DIST_SAMPLER_ARG="--use_dist_sampler"
else
    DIST_SAMPLER_ARG=""
fi

# Prepare the use_batchnorm argument
if [ "$USE_BATCHNORM" = "True" ]; then
    BATCHNORM_ARG="--use_batchnorm"
else
    BATCHNORM_ARG=""
fi

if [ "$USE_ORIG_PARAMS" = "True" ]; then
    ORIG_PARAMS_ARG="--use_batchnorm"
else
    ORIG_PARAMS_ARG=""
fi

if [ "$USE_SYNC" = "True" ]; then
    SYNC_ARG="--use_batchnorm"
else
    SYNC_ARG=""
fi

if [ "$USE_COMPILE" = "True" ]; then
    COMPILE_ARG="--compile"
else
    COMPILE_ARG=""
fi

# Generate SBATCH directives
cat << EOF > temp-run-mnist.sh
#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=v100:$GPUS
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH -o mnist-logs/mnist_%j.out
#SBATCH -e mnist-logs/mnist_%j.err
#SBATCH --time=00:30:00

source activate mdist-mmidas

python fsdp_mnist.py --epochs $EPOCHS --batch-size $BATCH_SIZE --model $MODEL --gpus $GPUS --num_workers 4 --sharding $STRAT $DIST_SAMPLER_ARG $BATCHNORM_ARG $ORIG_PARAMS_ARG $SYNC_ARG $COMPILE_ARG --mixed $MIXED
EOF

# Submit the job
sbatch temp-run-mnist.sh