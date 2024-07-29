import argparse
import builtins
import datetime
from functools import partial, reduce
from operator import mul
import os
from pathlib import Path
import random
from time import sleep
import string

import numpy as np
from pyrsistent import PMap, PVector
def set_call_(cls, fun):
    cls.__call__ = fun
set_call_(PMap, lambda self, x: self[x])
set_call_(PVector, lambda self, x: self[x])
from pyrsistent import pmap, m, pvector, v, pset, s
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, CPUOffload)
from torch.distributed.fsdp.wrap import (enable_wrap,
                                         size_based_auto_wrap_policy, wrap)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import wandb

from mmidas.cpl_mixvae import cpl_mixVAE, is_master
from mmidas.utils.tools import get_paths
from mmidas.utils.dataloader import (load_data, get_loaders, din,
                                     c_onehot_data, c_p_data, make_data, make_loaders)


from pyrsistent import PMap, PVector
def set_call_(cls, fun):
    cls.__call__ = fun
set_call_(PMap, lambda self, x: self[x])
set_call_(PVector, lambda self, x: self[x])

from pyrsistent import pmap, m, pvector, v

def is_path(x):
    return isinstance(x, Path)

def wrap_in_path(x):
    if is_path(x):
        return x
    else:
        return Path(x)

def using_a100():
    return is_a100(current_gpu())

def is_a100(gpu):
    return 'a100' in lower(gpu)

def lower(x):
    return x.lower()

def current_gpu():
    return torch.cuda.get_device_name()

def count_gpus():
  return torch.cuda.device_count()

# TODO: make this a generic function
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# TODO: change this to sizeof
def sizeof(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

# TODO: change this to sizeof
def print_sizeof(model):
    print(f"size: {sizeof(model):.2f} MB")

def setup_print_(rank):
  builtins.print = partial(print, f"[rank {rank}]")

def print_summary(model, rank):
    if is_master(rank):
        print(f"params: {count_params(model)}")
        print_sizeof(model)
        print(f"{model}")

# TODO: add a100 environmental flags
def set_environ_flags_():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(12355)
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_LOGS"]="+dynamo"
    os.environ["TORCHDYNAMO_VERBOSE"]="1"
    os.environ.update({
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    })
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    if is_a100(current_gpu()):
        print(f"using a100..")
        set_a100_flags_()

def set_a100_flags_():
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

def make_timeout(s):
  return datetime.timedelta(seconds=s)

def setup_process_group_(rank, world_size):
   dist.init_process_group('nccl', rank=rank, world_size=world_size, 
                           timeout=make_timeout(180))
  
def cleanup_process_group_():
    dist.destroy_process_group()

def setup_distributed_(rank, world_size):
  set_environ_flags_()
  setup_process_group_(rank, world_size)

def cleanup_distributed_():
  dist.destroy_process_group()

def is_imported(m):
  return m in globals()

# these three functions have three separate random APIs, which SUCK!
def set_random_seed_(seed):
    if is_imported('random'):
        random.seed(seed)

def set_numpy_seed_(seed):
    if is_imported('np'):
        np.random.seed(seed)

def set_torch_seed_(seed):
    if is_imported('torch'):
        torch.manual_seed(seed)

def set_seed_(seed):
    set_random_seed_(seed)
    set_numpy_seed_(seed)
    set_torch_seed_(seed)

def make_args(parser, **kwargs):
    return parser.parse_args()
    return pmap({
        'n_categories': kwargs.get('n_categories', 120),
        'state_dim': kwargs.get('state_dim', 2),
        'n_arm': kwargs.get('n_arm', 2),
        'temp': kwargs.get('temp', 1),
        'tau': kwargs.get('tau', .005),
        'beta': kwargs.get('beta', .01),
        'lam': kwargs.get('lam', 1),
        'lam_pc': kwargs.get('lam_pc', 1),
        'latent_dim': kwargs.get('latent_dim', 10),
        'n_epoch': kwargs.get('n_epoch', 10000),
        'n_epoch_p': kwargs.get('n_epoch_p', 10000),
        'min_con': kwargs.get('min_con', .99),
        'max_prun_it': kwargs.get('max_prun_it', 50),
        'ref_pc': kwargs.get('ref_pc', False),
        'fc_dim': kwargs.get('fc_dim', 100),
        'batch_size': kwargs.get('batch_size', 5000),
        'variational': kwargs.get('variational', True),
        'augmentation': kwargs.get('augmentation', False),
        'lr': kwargs.get('lr', .001),
        'p_drop': kwargs.get('p_drop', 0.5),
        's_drop': kwargs.get('s_drop', 0.2),
        'pretrained_model': kwargs.get('pretrained_model', False),
        'n_pr': kwargs.get('n_pr', 0),
        'loss_mode': kwargs.get('loss_mode', 'MSE'),
        'n_run': kwargs.get('n_run', 1),
        'hard': kwargs.get('hard', False),
        'device': kwargs.get('device', None),
    })

def make_data_dir(config):
    return config['paths']['main_dir'] / config['paths']['data_path']

def get_saving_path(config):
    return config['paths']['main_dir'] / config['paths']['saving_path']

def make_data_file(sub_file, config):
    return make_data_dir(config) / config[sub_file]['anndata_file']

def make_folder_name(n_run, n_categories, state_dim, augmentation, lr, n_arm, 
                     batch_size, n_epoch, n_epoch_p):
    return f'run_{n_run}_K_{n_categories}_Sdim_{state_dim}_aug_{augmentation}_lr_{lr}_n_arm_{n_arm}_nbatch_{batch_size}' + \
           f'_train.ipynb_nepoch_{n_epoch}_nepochP_{n_epoch_p}'

def make_saving_dir(folder_name, config):
    return get_saving_path(config) / folder_name

def make_dirs_(name):
    os.makedirs(name, exist_ok=True)

def make_file_config(toml_file, sub_file):
    return get_paths(toml_file=toml_file, sub_file=sub_file)

def main_dir_file(config):
    return config['paths']['main_dir']

def make_aug_file(toml_file, sub_file):
    config = make_file_config(toml_file, sub_file)
    return main_dir_file(config) / config[sub_file]['aug_model']
    
def make_trained_model_file(config, sub_file):
    return config['paths']['main_dir'] / config[sub_file]['trained_model']

def is_smartseq(dataset):
    return dataset == 'smartseq'
    
def make_wrap_policy(params):
    return partial(size_based_auto_wrap_policy, min_num_params=params)

def fsdp(*args, **kwargs):
  return FSDP(*args, **kwargs)

def get_batch_size(args):
    return args.batch_size

def is_parallel(world_size):
    return world_size > 1

def is_args(x):
    return isinstance(x, argparse.Namespace)

def use_fsdp(x):
    if is_args(x):
        return x.fsdp and is_parallel(count_gpus_args(x)) and not (get_device(x) == 'cpu' or get_device(x) == 'mps')
    else:
        raise ValueError("type x not supported")

def get_device(x):
    if is_args(x):
        return x.device
    else:
        raise ValueError("type x not supported")
    
def use_compile(x):
    return x.compile

def use_dist_sampler(args):
    if is_args(args):
        return args.use_dist_sampler
    else:
        raise ValueError("type x not supported")

# TODO: can make this generic
def count_gpus_args(args):
    if args.gpus == -1:
        return count_gpus()
    else:
        return args.gpus
    
def use_augmentation(args):
    return args.augmentation

def tag_(x, tag):
    x.tag = tag

def set_wandb_flags_():
    wandb.require('service')
    wandb.require('core')
    
def make_logger_config(**kwargs):
    return pmap(kwargs)

def make_logger(project, config={}, group_name=None):
    set_wandb_flags_()
    if not group_name:
        group = next_group_name_wandb(project)
    else:
        group = group_name
    print(f"group: {group}")
    run = wandb.init(project=project, group=group, config=dict(config))
    wandb.define_metric('epoch')
    wandb.define_metric('avg_rec_loss', step_metric='epoch')
    def log(metrics, **kwargs):
        run.log(metrics, **kwargs)
    tag_(log, 'wandb')

    def log_cleanup_():
        run.finish()
    return log, log_cleanup_

def type_logger(logger):
    return logger.tag 

def is_wandb_logger(logger):
    return logger.tag == 'wandb'

def project_path_wandb(project, username):
    return f"{username}/{project}"

def make_api_wandb():
    return wandb.Api()

def is_api_wandb(api):
    return isinstance(api, wandb.Api)

def make_path_wandb(project, entity=None):
    if is_entity_wandb(entity):
        return f"{entity}/{project}"
    else:
        return make_path_wandb(project, current_entity_wandb())

def projects_wandb(entity=None):
    api = make_api_wandb()
    if is_entity_wandb(entity):
        return api.projects(entity)
    else:
        return projects_wandb(current_entity_wandb())

def runs_wandb(project, entity=None):
    api = make_api_wandb()
    try:
        runs = api.runs(make_path_wandb(project, entity))
        return pvector([run for run in runs])
    except Exception as e:
        return pvector([])

def current_entity_wandb():
    api = make_api_wandb()
    return api.default_entity

def is_entity_wandb(entity):
    return isinstance(entity, str)

def groups_wandb(project, entity=None):
    if is_entity_wandb(entity):
        runs = runs_wandb(project, entity)
    else:
        runs = runs_wandb(project, current_entity_wandb())
    return pset({run.group for run in runs if run.group is not None})

def count_groups_wandb(project, entity=None):
    return len(groups_wandb(project, entity))

def next_group_wandb(project, entity=None):
    return count_groups_wandb(project, entity)

def next_group_name_wandb(project, entity=None):
    return f"group_{next_group_wandb(project, entity)}"

# TODO
def del_group_wandb_(group, project, entity=None):
    api = make_api_wandb()
    api.runs(make_path_wandb(project, entity)).delete(query=f"group:{group}")

def set_get_id_(args, id):
    args.id = id

def get_id(args):
    return args.id

def random_string(n):
  return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def count_arms(args):
    return args.n_arm

def load_model(file, device='cpu'):
    return torch.load(file, map_location=device)

def use_mixed(args):
    return args.mixed

def set_group_name_(args, name):
    args.group_name = name

def get_group_name(args):
    return args.group_name

def fsdp_main(rank, world_size, args):
    setup_print_(rank)
    print(f"starting...")

    setup_distributed_(rank, world_size)

    toml_file = 'pyproject.toml'
    sub_file = 'mouse_smartseq'
    config = get_paths(toml_file=toml_file, sub_file=sub_file)
    data_file = (Path(config[sub_file]['data_path']) 
                 / Path(config[sub_file]['anndata_file']))
    folder_name = make_folder_name(args.n_run, args.n_categories, args.state_dim, args.augmentation,
                                   args.lr, args.n_arm, args.batch_size, args.n_epoch, args.n_epoch_p)
    saving_folder = config['paths']['main_dir'] / config[sub_file]['saving_path']
    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)
    if args.augmentation:
        aug_file = config['paths']['main_dir'] / config[sub_file]['aug_model']
    else:
        aug_file = ''

    if args.pretrained_model:
        trained_model = config['paths']['main_dir'] / config[sub_file]['trained_model']
    else:
        trained_model = ''

    data = load_data(datafile=data_file)


    cplMixVAE = cpl_mixVAE(saving_folder=saving_folder,
                           device=rank,
                           aug_file=aug_file,
                           load_weights=args.load_weights)
    
    fold = 0
    loaders = cplMixVAE.get_dataloader(dataset=data['log1p'],
                                       label=data['cluster'],
                                       batch_size=args.batch_size,
                                       n_aug_smp=0,
                                       fold=fold,
                                       rank=rank,
                                       world_size=world_size,
                                       use_dist_sampler=use_dist_sampler(args))
    _, train_loader, _, test_loader = loaders

    cplMixVAE.init_model(n_categories=args.n_categories,
                         state_dim=args.state_dim,
                         input_dim=din(data),
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
                         trained_model='',
                         n_pr=args.n_pr,
                         mode=args.loss_mode)
    
    cplMixVAE.model = cplMixVAE.model.to(rank)
    if use_fsdp(args):
        print('using fsdp...')
        if use_mixed(args):
            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            )
        else:
            mp = None
        cplMixVAE.model = fsdp(cplMixVAE.model, auto_wrap_policy=make_wrap_policy(20000),
                            use_orig_params=True, device_id=rank, sharding_strategy=ShardingStrategy.NO_SHARD,
                            mixed_precision=mp)
    
    if use_compile(args):
        print('compiling model...')
        cplMixVAE.model = torch.compile(cplMixVAE.model)
    
    # TODO: test loss all reduce
    lc = make_logger_config(use_dist_sampler=use_dist_sampler(args),
                                n_epoch=args.n_epoch,
                                use_fsdp=use_fsdp(args),
                                world_size=world_size,
                                arms=count_arms(args),
                                augmentation=use_augmentation(args))
    
    log, cleanup_log_ = make_logger('mmidas', config=lc, group_name=get_group_name(args))

    
    model_file = cplMixVAE.train_(train_loader=train_loader,
                                    test_loader=test_loader,
                                    n_epoch=args.n_epoch,
                                    n_epoch_p=args.n_epoch_p,
                                    c_onehot=c_onehot_data(data),
                                    c_p=c_p_data(data),
                                    min_con=args.min_con,
                                    max_prun_it=args.max_prun_it,
                                    rank=rank,
                                    world_size=world_size,
                                    log=log)
        
    
    print_summary(cplMixVAE.model, rank)
    if use_augmentation(args) and is_master(rank):
        print(f"augmenter params: {count_params(cplMixVAE.netA)}")
        print_sizeof(cplMixVAE.netA)
    cleanup_log_()
    cleanup_distributed_()

def spawn_(fun, procs, args):
    mp.spawn(fun, args=(procs, args), nprocs=procs, join=True)

# Run the main function when the script is executed
if __name__ == "__main__":
    # Setup argument parser for command line arguments
    parser = argparse.ArgumentParser()

    # Define command line arguments
    parser.add_argument("--n_categories", default=120, type=int, help="(maximum) number of cell types")
    parser.add_argument("--state_dim", default=2, type=int, help="state variable dimension")
    parser.add_argument("--n_arm", default=2, type=int, 
                        help="number of mixVAE arms for each modality")
    parser.add_argument("--temp", default=1, type=float, help="gumbel-softmax temperature")
    parser.add_argument("--tau", default=.005, type=float, help="softmax temperature")
    parser.add_argument("--beta", default=.01, type=float, help="KL regularization parameter")
    parser.add_argument("--lam", default=1, type=float, help="coupling factor")
    parser.add_argument("--lam_pc", default=1, type=float, help="coupling factor for ref arm")
    parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
    parser.add_argument("--n_epoch", default=10, type=int, help="Number of epochs to train")
    parser.add_argument("--n_epoch_p", default=10000, type=int, help="Number of epochs to train pruning algorithm")
    parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
    parser.add_argument("--max_prun_it", default=50, type=int, help="maximum number of pruning iterations")
    parser.add_argument("--ref_pc", default=False, type=bool, help="use a reference prior component")
    parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
    parser.add_argument("--batch_size", default=5000, type=int, help="batch size")
    parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
    parser.add_argument("--augmentation", default=False, action='store_true', help="enable VAE-GAN augmentation")
    parser.add_argument("--load_weights", default=False, action='store_true', help="load previous augmentation weights")
    # parser.add_argument("--augmentation", default=False, type=bool, help="enable VAE-GAN augmentation")
    parser.add_argument("--lr", default=.001, type=float, help="learning rate")
    parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
    parser.add_argument("--s_drop", default=0.2, type=float, help="state probability of dropout")
    parser.add_argument("--pretrained_model", default=False, type=bool, help="use pretrained model")
    parser.add_argument("--n_pr", default=0, type=int, help="number of pruned categories in case of using a pretrained model")
    parser.add_argument("--loss_mode", default='MSE', type=str, help="loss mode, MSE or ZINB")
    parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
    parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
    parser.add_argument("--device", default='cuda', type=str, help="computing device, either 'cpu' or 'cuda'.")
    parser.add_argument("--gpus", default=-1, type=int, help="number of gpus")
    parser.add_argument("--fsdp", default=False, action='store_true', help="use fsdp")
    parser.add_argument("--use_dist_sampler", default=False, action='store_true', help="use distributed sampler")
    parser.add_argument("--compile", action='store_true', default=False, help='torch.compile the model')
    parser.add_argument("--mixed", action="store_true", default=False,
                        help="enable bf16 mixed precision training")


    args = make_args(parser)
    # main(**vars(args))
    set_group_name_(args, next_group_name_wandb('mmidas'))
    world_size = count_gpus_args(args)
    spawn_(fsdp_main, world_size, args)

# normal mmidas model per epoch: 


# max arms
# smartseq: 
    # no fsdp: 42, 43M params, 164MB
    # fsdp: 


# automatically adjust dataloader configuration based on available cpus 

# jointly learning
    # maybe attention
    # maybe contrastive learning
        # metric for who is close and who is far to include in the consensus measure

# 164416304.0000
# 235004384.0000

# 100 epochs
    # sampler: 2083725.3750
    # no sampler: 931459
    # no sampler, no fsdp


# data 
#  1. transcriptomic
#  2. time series( electrophysiology)
#  3. morphological (arbor density)
#  4. rna seq


# TODO:
    # [] add a100 optimizations
    # [] lscpu for num threads
    # [] add back adist sampler
    # [] 5 horizontal mmidas layers, add noise, then 5 more horizontal mmidas layers?