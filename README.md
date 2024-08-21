# Dist_MMIDAS
Distributed training of multi-arm variational autoencoding networks

## installation

to recreate the conda environment used for this project, do this: 

1. Clone the repo
```
git clone https://github.com/AllenInstitute/Dist_MMIDAS.git
cd Dist_MMIDAS
```
2. Install `torch` with CUDA >= 2.0 and `tqdm`:

You can either recreate the exact conda environment we used for this project (which likely has more packages than you actually need) by:
```
conda env create -f environment.yml
```
or, just follow the standard instructions for installing `torch` with CUDA >= 2.0 on your machine, as well as `tqdm`. 

3. activate the environment
```
conda activate your_environment_name
```