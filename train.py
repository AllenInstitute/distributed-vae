import argparse
import os
import random
import signal
from copy import deepcopy
from itertools import starmap
from pathlib import Path

import numpy as np
import torch as th
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.distributed.fsdp import BackwardPrefetch, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType

import fsdp_mnist as utils
import wandb
from mmidas._dist_utils import set_print, init_dist_env
from mmidas._utils import mapv
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.nn_model import mixVAE_model
from mmidas.utils.dataloader import get_loaders, load_data
from mmidas.utils.tools import get_paths

SEED = 546

def parse_toml(toml_file: str, sub_file: str, args=None, trained=False):
    def count_existing(saving_folder, acc=0):
        if not os.path.exists(saving_folder + f"_RUN{acc}"):
            return acc
        else:
            return count_existing(saving_folder, acc + 1)

    def mk_saving_folder(saving_folder, run):
        return saving_folder + f"_RUN{run}"

    config = get_paths(toml_file=toml_file, sub_file=sub_file)
    data_file = Path(config[sub_file]["data_path"]) / Path(
        config[sub_file]["anndata_file"]
    )
    folder_name = (
        f"K{args.n_categories}_S{args.state_dim}_AUG{args.augmentation}_LR{args.lr}_A{args.n_arm}_B{args.batch_size}"
        + f"_E{args.n_epoch}_Ep{args.n_epoch_p}"
    )
    saving_folder = str(
        config["paths"]["main_dir"] / config[sub_file]["saving_path"] / folder_name
    )
    return dict(
        mapv(
            str,
            {
                "data": data_file,
                "saving": mk_saving_folder(
                    saving_folder, count_existing(saving_folder)
                ),
                "aug": config["paths"]["main_dir"] / config[sub_file]["aug_model"],
                "trained": config["paths"]["main_dir"]
                / config[sub_file]["trained_model"]
                if trained
                else "",
            }.items(),
        )
    )


def main(rank, ws, args):
    if ws > 1:
        init_dist_env(rank, ws, args.addr, args.port)

    # Load configuration paths
    files = parse_toml("mmidas.toml", "mouse_smartseq", args, trained=False)
    print(f" -- making folders: {files['saving']} -- ")
    os.makedirs(files["saving"], exist_ok=True)
    os.makedirs(files["saving"] + "/model", exist_ok=True)

    # Load data
    data = load_data(datafile=files["data"])
    (N, D) = data["log1p"].shape
    print(f"# cells: {N}, # genes: {D}")

    # Initialize the coupled mixVAE (MMIDAS) model
    cplMixVAE = cpl_mixVAE(files["saving"], files["aug"], rank)

    # Make data loaders for training, validation, and testing
    fold = 0  # fold index for cross-validation, for reproducibility purpose
    train_loader, test_loader, _ = get_loaders(
        dataset=data["log1p"],
        seed=SEED,
        batch_size=args.batch_size,
        world_size=ws,
        rank=rank,
        use_dist_sampler=args.use_dist_sampler,
    )

    # Initialize the model with specified parameters
    cplMixVAE.init_model(
        n_categories=args.n_categories,
        state_dim=args.state_dim,
        input_dim=D,
        fc_dim=args.fc_dim,
        lowD_dim=args.latent_dim,
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
        ref_prior=args.ref_pc,
        variational=args.variational,
        trained_model=files["trained"],
        n_pr=args.n_pr,
        mode=args.loss_mode,
    )

    # Train and save the model
    if args.use_wandb:
        run = wandb.init(project="mmidas-experiments", config=vars(args))
    else:
        run = None

    if ws > 1:
        cplMixVAE.model = FSDP(
            cplMixVAE.model, auto_wrap_policy=utils.make_wrap_policy(20000)
        )
    if args.optimizer == "adam":
        cplMixVAE.optimizer = optim.Adam(cplMixVAE.model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        cplMixVAE.optimizer = optim.AdamW(cplMixVAE.model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f"optimizer {args.optimizer} not implemented")

    cplMixVAE.train(
        train_loader=train_loader,
        test_loader=test_loader,
        n_epoch=args.n_epoch,
        n_epoch_p=args.n_epoch_p,
        c_onehot=data["c_onehot"],
        c_p=data["c_p"],
        min_con=args.min_con,
        max_prun_it=args.max_prun_it,
        run=run,
        ws=ws,
        rank=rank,
    )

    if ws > 1:
        dist.destroy_process_group()


# Run the main function when the script is executed
if __name__ == "__main__":
    # Setup argument parser for command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_arm", default=2, type=int, help="number of mixVAE arms for each modality"
    )

    parser.add_argument(
        "--n_categories", default=92, type=int, help="number of cell types"
    )
    parser.add_argument(
        "--state_dim", default=2, type=int, help="state variable dimension"
    )
    parser.add_argument(
        "--temp", default=1, type=float, help="gumbel-softmax temperature"
    )
    parser.add_argument("--tau", default=0.005, type=float, help="softmax temperature")
    parser.add_argument(
        "--beta", default=1, type=float, help="KL regularization parameter"
    )
    parser.add_argument("--lam", default=1, type=float, help="coupling factor")
    parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
    parser.add_argument(
        "--n_epoch", default=50000, type=int, help="Number of epochs to train"
    )
    parser.add_argument(
        "--n_epoch_p",
        default=0,
        type=int,
        help="Number of epochs to train pruning algorithm",
    )
    parser.add_argument("--min_con", default=0.99, type=float, help="minimum consensus")
    parser.add_argument(
        "--max_prun_it",
        default=0,
        type=int,
        help="minimum number of samples in a class",
    )
    parser.add_argument(
        "--fc_dim", default=100, type=int, help="number of nodes at the hidden layers"
    )
    parser.add_argument("--batch_size", default=5000, type=int, help="batch size")
    parser.add_argument(
        "--variational", default=True, type=bool, help="enable variational mode"
    )
    parser.add_argument(
        "--augmentation", default=True, type=bool, help="enable VAE-GAN augmentation"
    )
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--p_drop", default=0.5, type=float, help="input probability of dropout"
    )
    parser.add_argument(
        "--s_drop", default=0.0, type=float, help="state probability of dropout"
    )
    parser.add_argument(
        "--lam_pc", default=1, type=float, help="coupling factor for ref arm"
    )
    parser.add_argument(
        "--ref_pc", default=False, type=bool, help="use a reference prior component"
    )
    parser.add_argument(
        "--pretrained_model", default=False, type=bool, help="use pretrained model"
    )
    parser.add_argument(
        "--n_pr",
        default=0,
        type=int,
        help="number of pruned categories in case of using a pretrained model",
    )
    parser.add_argument(
        "--loss_mode", default="MSE", type=str, help="loss mode, MSE or ZINB"
    )
    parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
    parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
    parser.add_argument(
        "--dataset",
        default="mouse_smartseq",
        type=str,
        help="dataset name, e.g., 'mouse_smartseq', 'mouse_ctx_10x'",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="computing device, either 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--use-wandb", default=False, action="store_true", help="use wandb for logging"
    )
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--use_orig_params", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_dist_sampler", default=False, action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="adam")
    args = parser.parse_args()

    ws = args.gpus  # world size
    num_workers = args.num_workers

    print(f"world size: {ws}")
    args.num_workers = num_workers
    if ws > 1:
        raise NotImplementedError("distributed training is not yet supported")
        addr = utils.find_addr()
        port = utils.find_port(addr)
        prefetch_factor = args.prefetch_factor
        if prefetch_factor is None:
            prefetch_factor = utils.get_prefetch_factor()

        args.addr = addr
        args.port = port
        args.prefetch_factor = prefetch_factor
        print(args)
        mp.spawn(main, args=(ws, args), nprocs=ws, join=True)
    else:
        main(args.device, 1, args)
