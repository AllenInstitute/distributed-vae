import argparse
from copy import deepcopy
import os
import numpy as np
import signal
import torch as th
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
import random
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.tools import get_paths
from mmidas.utils.dataloader import load_data, get_loaders
from pathlib import Path
from pyrsistent import pmap

from mmidas.nn_model import mixVAE_model

import wandb

import fsdp_mnist as utils

signal.signal(signal.SIGINT, lambda _, __: dist.destroy_process_group())

def parse_toml_(toml_file: str, sub_file: str, args=None, trained=False):
    def _make_saving_folders_(saving_folder, existing=0):
        if not os.path.exists(saving_folder + f'_RUN{existing}'):
            return saving_folder + f'_RUN{existing}'
        else:
            return _make_saving_folders_(saving_folder, existing + 1)
    
    config = get_paths(toml_file=toml_file, sub_file=sub_file)
    data_file = Path(config[sub_file]['data_path']) / Path(config[sub_file]['anndata_file'])
    folder_name = f'K{args.n_categories}_S{args.state_dim}_AUG{args.augmentation}_LR{args.lr}_A{args.n_arm}_B{args.batch_size}' + \
                    f'_E{args.n_epoch}_Ep{args.n_epoch_p}'
    saving_folder = config['paths']['main_dir'] / config[sub_file]['saving_path'] / folder_name
    return pmap(map(lambda kv: (kv[0], str(kv[1])), {
        'data': data_file,
        'saving': _make_saving_folders_(str(saving_folder)),
        'aug': config['paths']['main_dir'] / config[sub_file]['aug_model'],
        'trained': config['paths']['main_dir'] / config[sub_file]['trained_model'] if trained else ''
    }.items()))
    

def update_args(args, **kw):
    cpy = deepcopy(args)
    for k, v in kw.items():
        setattr(cpy, k, v)
    return cpy

# Main function
def main(r, ws, args):
    if ws > 1:
        utils.set_prn_(r)
        utils.su_dist_(r, ws, args.addr, args.port)
        th.cuda.set_device(r)

    seed = 546
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Load configuration paths
    cfg = parse_toml_('pyproject.toml', 'mouse_smartseq', args, trained=False)
    print(f' -- making folders: {cfg.saving} -- ')
    os.makedirs(cfg.saving, exist_ok=True)
    os.makedirs(cfg.saving + '/model', exist_ok=True)

    # Load data
    data = load_data(datafile=cfg.data)
    print(f"# cells: {data['log1p'].shape[0]}, # genes: {data['log1p'].shape[1]}")

    # Initialize the coupled mixVAE (MMIDAS) model
    cplMixVAE = cpl_mixVAE(saving_folder=cfg.saving,
                                 device=r,
                                 aug_file=cfg.aug)

    # Make data loaders for training, validation, and testing
    fold = 0 # fold index for cross-validation, for reproducibility purpose
    train_loader, test_loader, alldata_loader = get_loaders(dataset=data['log1p'],
                                                            seed = seed,
                                                            batch_size=args.batch_size,
                                                            world_size=ws,
                                                            rank=r,
                                                            use_dist_sampler=args.use_dist_sampler)

    # Initialize the model with specified parameters
    cplMixVAE.init_model(n_categories=args.n_categories,
                         state_dim=args.state_dim,
                         input_dim=data['log1p'].shape[1],
                         fc_dim=args.fc_dim,
                         lowD_dim=args.latent_dim, # <-- good programming
                         x_drop=args.p_drop,
                         s_drop=args.s_drop,
                         lr=args.lr,
                         n_arm=args.n_arm,
                         temp=args.temp,
                         hard=args.hard,
                         tau=args.tau,
                         lam=args.lam,
                         lam_pc=args.lam_pc,
                         beta=args.beta,
                         ref_prior=args.ref_pc, # <-- good programming
                         variational=args.variational,
                         trained_model=cfg.trained,
                         n_pr=args.n_pr,
                         mode=args.loss_mode) # <-- good programming

    # Train and save the model
    run = wandb.init(project='mmidas-experiments', config=vars(args)) if args.use_wandb else None
    cplMixVAE.model = (FSDP(cplMixVAE.model, auto_wrap_policy=utils.make_wrap_policy(20000)) 
                       if ws > 1 else cplMixVAE.model)
    cplMixVAE.optimizer = th.optim.Adam(cplMixVAE.model.parameters(), lr=args.lr)

    model_file = cplMixVAE.train(train_loader=train_loader,
                                 test_loader=test_loader,
                                 n_epoch=args.n_epoch,
                                 n_epoch_p=args.n_epoch_p,
                                 c_onehot=data['c_onehot'],
                                 c_p=data['c_p'],
                                 min_con=args.min_con,
                                 max_prun_it=args.max_prun_it,
                                 run=run, ws=ws, rank=r)

    if ws > 1:
        dist.destroy_process_group()

# Run the main function when the script is executed
if __name__ == "__main__":
    # Setup argument parser for command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_arm", default=2, type=int, help="number of mixVAE arms for each modality")

    parser.add_argument("--n_categories", default=92, type=int, help="number of cell types")
    parser.add_argument("--state_dim", default=2, type=int, help="state variable dimension")
    parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
    parser.add_argument("--tau",  default=.005, type=float, help="softmax temperature")
    parser.add_argument("--beta",  default=1, type=float, help="KL regularization parameter")
    parser.add_argument("--lam",  default=1, type=float, help="coupling factor")
    parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
    parser.add_argument("--n_epoch", default=50000, type=int, help="Number of epochs to train")
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
    parser.add_argument("--device", default='cuda', type=str, help="computing device, either 'cpu' or 'cuda'.")
    parser.add_argument("--use-wandb", default=False, action='store_true', help="use wandb for logging")
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--use_orig_params', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--use_dist_sampler', default=False, action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=-1)
    args = utils.make_args(parser)
    
    ws = utils.ct_gpu_args(args)
    print(f'ws: {ws}')
    if ws > 1:
        addr = utils.get_free_addr()
        args = update_args(args, addr=addr, 
                           port=utils.get_free_port(addr), 
                           num_workers=utils.count_num_workers(args),
                           gpus=ws, prefetch_factor=utils.get_prefetch_factor(args))
        print(args)
        mp.spawn(main, args=(ws, args), nprocs=ws, join=True)
    else:
        main(args.device, 1, args)

# TODO:
    # [] fix th.compile for mps
    # [] check reparam_trick()