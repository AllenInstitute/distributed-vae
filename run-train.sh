#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <arms>"
    exit 1
fi

ARMS=$1

cat << EOF > temp-run-train.sh
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

python train.py --n_arms $ARMS --n_epoch 3
EOF

sbatch temp-run-train.sh