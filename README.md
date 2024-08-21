# distributed-vae
Distributed training of multi-arm variational autoencoding networks. This repository contains 
- experiment code for training multi-arm variational autoencoding networks with the fully-sharded data parallelism (FSDP) strategy in PyTorch
- a tutorial on training using FSDP with some simple models on the MNIST dataset

## Installation

To recreate the conda environment used for this project,: 

1. Clone the repo
```
git clone https://github.com/AllenInstitute/distributed-vae.git
cd Dist_MMIDAS
```
2. Install `torch` with CUDA >= 2.0 and `tqdm`:

You can either recreate the exact conda environment we used for this project (which likely has more packages than you actually need) by:
```
conda env create -f environment.yml
```
or, just follow the standard instructions for installing `torch` with CUDA >= 2.0 on your machine, as well as `tqdm`. 

3. Activate the environment
```
conda activate your_environment_name
```

## Quick start

The most important part of this repository are the two files `fsdp_tutorial.ipynb` and `fsdp_tutorial.py`. 

The file `fdsp_tutorial.ipynb` is a tutorial that walks through step-by-step on how to use the FSDP training strategy in PyTorch. This is likely what you are looking for. Activate your conda environment (instructions above) and walk through this notebook to learn how to use FSDP in PyTorch.

The file `fsdp_tutorial.py` is a Python script containing the same code as the tutorial notebook. This is suitable for running the tutorial code as a job on HPC cluster environments (such as SLURM).

## TODO
- [ ] `sbatch` script for running `fsdp_tutorial.py` on SLURM
- [ ] Cleanup files in `dist/` directory

## QR Code
![QR Code](./qr-code.png)
