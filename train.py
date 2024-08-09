import argparse
import os
import numpy as np
import signal
import torch as th
import random
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.tools import get_paths
from mmidas.utils.dataloader import load_data, get_loaders
from pathlib import Path

import wandb

import fsdp_mnist as fs

def is_path(x):
    return isinstance(x, Path)

def wrap_in_path(x):
    return wrap_in_path(Path(x)) if not is_path(x) else x

signal.signal(signal.SIGINT, lambda _, __: fs.cu_dist_())

# Main function
def main(r, ws, args):
    globals().update(vars(args))

    if ws > 1:
        fs.set_prn_(r)
        fs.su_dist_(r, ws, addr, port)
        fs.set_gpu_(r)

    seed = 546
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Load configuration paths
    toml_file = 'pyproject.toml'
    config = get_paths(toml_file=toml_file, sub_file=dataset)
    data_file = wrap_in_path(config[dataset]['data_path']) / wrap_in_path(config[dataset]['anndata_file'])   

    # Define folder name for saving results
    folder_name = f'Run{n_run}_K{n_categories}_S{state_dim}_AUG{augmentation}_LR{lr}_A{n_arm}_B{batch_size}' + \
                  f'_E{n_epoch}_Ep{n_epoch_p}'
    saving_folder = config['paths']['main_dir'] / config[dataset]['saving_path'] / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)
    

    # Determine augmentation file path
    if augmentation:
        aug_file = config['paths']['main_dir'] / config[dataset]['aug_model']
    else:
        aug_file = ''

    # Determine pretrained model file path
    if pretrained_model:
        trained_model = config['paths']['main_dir'] / config[dataset]['trained_model']
    else:
        trained_model = ''

    # Load data
    data_dict = load_data(datafile=data_file)
    print("Data loaded successfully!")
    print(f"Number of cells: {data_dict['log1p'].shape[0]}, Number of genes: {data_dict['log1p'].shape[1]}")

    # Initialize the coupled mixVAE (MMIDAS) model
    cplMixVAE = cpl_mixVAE(saving_folder=saving_folder,
                                 device=r,
                                 aug_file=aug_file)

    # Make data loaders for training, validation, and testing
    fold = 0 # fold index for cross-validation, for reproducibility purpose
    alldata_loader, train_loader, validation_loader, test_loader = cplMixVAE.get_dataloader(dataset=data_dict['log1p'],
                                                                                             label=data_dict['cluster'],
                                                                                             batch_size=batch_size,
                                                                                             n_aug_smp=0,
                                                                                             fold=fold,
                                                                                             deterministic=True,
                                                                                             world_size=ws,
                                                                                             use_dist_sampler=use_dist_sampler,
                                                                                             rank=r)

    # Initialize the model with specified parameters
    cplMixVAE.init_model(n_categories=n_categories,
                         state_dim=state_dim,
                         input_dim=data_dict['log1p'].shape[1],
                         fc_dim=fc_dim,
                         lowD_dim=latent_dim,
                         x_drop=p_drop,
                         s_drop=s_drop,
                         lr=lr,
                         n_arm=n_arm,
                         temp=temp,
                         hard=hard,
                         tau=tau,
                         lam=lam,
                         lam_pc=lam_pc,
                         beta=beta,
                         ref_prior=ref_pc,
                         variational=variational,
                         trained_model=trained_model,
                         n_pr=n_pr,
                         mode=loss_mode)

    # Train and save the model
    run = wandb.init(project='mmidas-arms', config=_args) if use_wandb else None
    cplMixVAE.model = fs.fsdp(cplMixVAE.model, auto_wrap_policy=fs.make_wrap_policy(20000)) if ws > 1 else cplMixVAE.model
    model_file = cplMixVAE.train(train_loader=train_loader,
                                 test_loader=test_loader,
                                 n_epoch=n_epoch,
                                 n_epoch_p=n_epoch_p,
                                 c_onehot=data_dict['c_onehot'],
                                 c_p=data_dict['c_p'],
                                 min_con=min_con,
                                 max_prun_it=max_prun_it,
                                 run=run, ws=ws, rank=r)
    if ws > 1:
        fs.cu_dist_()


# Run the main function when the script is executed
if __name__ == "__main__":
    # Setup argument parser for command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_arm", default=2, type=int, help="number of mixVAE arms for each modality")

    parser.add_argument("--n_categories", default=115, type=int, help="number of cell types")
    parser.add_argument("--state_dim", default=2, type=int, help="state variable dimension")
    parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
    parser.add_argument("--tau",  default=.005, type=float, help="softmax temperature")
    parser.add_argument("--beta",  default=1, type=float, help="KL regularization parameter")
    parser.add_argument("--lam",  default=1, type=float, help="coupling factor")
    parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
    parser.add_argument("--n_epoch", default=20000, type=int, help="Number of epochs to train")
    parser.add_argument("--n_epoch_p", default=0, type=int, help="Number of epochs to train pruning algorithm")
    parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
    parser.add_argument("--max_prun_it", default=0, type=int, help="minimum number of samples in a class")
    parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
    parser.add_argument("--batch_size", default=5000, type=int, help="batch size")
    parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
    parser.add_argument("--augmentation", default=True, type=bool, help="enable VAE-GAN augmentation")
    parser.add_argument("--lr", default=.001, type=float, help="learning rate")
    parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
    parser.add_argument("--s_drop", default=0.0, type=float, help="state probability of dropout")
    parser.add_argument("--lam_pc", default=1, type=float, help="coupling factor for ref arm")
    parser.add_argument("--ref_pc", default=False, type=bool, help="use a reference prior component")
    parser.add_argument("--pretrained_model", default=False, type=bool, help="use pretrained model")
    parser.add_argument("--n_pr", default=0, type=int, help="number of pruned categories in case of using a pretrained model")
    parser.add_argument("--loss_mode", default='MSE', type=str, help="loss mode, MSE or ZINB")
    parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
    parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
    parser.add_argument("--dataset", default='mouse_smartseq', type=str, help="dataset name, e.g., 'mouse_smartseq', 'mouse_ctx_10x'")
    parser.add_argument("--device", default='cpu', type=str, help="computing device, either 'cpu' or 'cuda'.")
    parser.add_argument("--use-wandb", default=False, action='store_true', help="use wandb for logging")
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--use_orig_params', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--use_dist_sampler', default=False, action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=-1)
    args = fs.make_args(parser)
    
    ws = fs.ct_gpu_args(args)
    fs.prn(f'ws: {ws}')
    if ws > 0:
        args.addr = fs.get_free_addr()
        args.port = fs.get_free_port(args.addr)
        args.num_workers = fs.count_num_workers(args)
        args.gpus = ws
        args.prefetch_factor = fs.get_prefetch_factor(args)
        fs.prn(args)
        fs.spawn_(main, ws, args)
    else:
        main(args.device, 1, args)

# TODO:
    # [] fix th.compile for mps
    # [] check reparam_trick()