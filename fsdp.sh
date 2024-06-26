#!/bin/bash

DEFAULT_GPU_COUNT=4
DEFAULT_EPOCHS=10
DEFAULT_CONDA_ENV="dist-mmidas"
DEFAULT_PARTITION="celltypes"
DEFAULT_TIMEOUT=120

OPTIONS=""
LONGOPTIONS=gpus:,epochs:,env:,partition:,timeout:

PARSED=$(getopt --options p: --longoptions $LONGOPTIONS --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
  exit 2
fi

eval set -- "$PARSED"

GPU_COUNT=$DEFAULT_GPU_COUNT
EPOCHS=$DEFAULT_EPOCHS
CONDA_ENV=$DEFAULT_CONDA_ENV
PARTITION=$DEFAULT_PARTITION
TIMEOUT=$DEFAULT_TIMEOUT

# Process the options
while true; do
  case "$1" in
    --gpus)
      GPU_COUNT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --env)
      CONDA_ENV="$2"
      shift 2
      ;;
    -p|--partition)
      PARTITION="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unexpected option: $1"
      exit 3
      ;;
  esac
done

SCRIPT="mnist.slurm"

cat <<EOF > $SCRIPT
#!/bin/bash
#SBATCH --job-name=fsdp_mnist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:$GPU_COUNT
#SBATCH --time=01:00:00
#SBATCH --output=slurm-log/out_%j.log
#SBATCH --error=slurm-log/error_%j.log
#SBATCH --partition=$PARTITION

mkdir -p slurm-log

export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=\$((\$SLURM_NNODES * \$SLURM_NTASKS_PER_NODE))

eval "\$(conda shell.bash hook)"
conda activate $CONDA_ENV

echo "running fsdp_mnist.py..."
srun python fsdp_mnist.py --multinode --backend nccl --epochs $EPOCHS --timeout $TIMEOUT
EOF

chmod +x $SCRIPT

sbatch $SCRIPT

echo "submitted job with script: $SCRIPT"
echo ""

cat $SCRIPT
# TODO: documentation