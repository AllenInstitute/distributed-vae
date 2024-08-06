


import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
_pth = '/allen/programs/celltypes/workgroups/mousecelltypes/Hilal/MMIDAS'
if _pth not in sys.path:
    sys.path.append(_pth)

import mmidas
from mmidas.nn_model import mixVAE_model, loss_fn
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.tools import get_paths
from mmidas.utils.dataloader import load_data, get_loaders

import functools
import torch
from contextlib import contextmanager
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import datetime
import wandb

# TODO: add way to use w/o contextmanager
@contextmanager
def dist_init(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=300))
  try:
    yield
  finally:
    dist.destroy_process_group()

# Define the main function
# [x] each arm should probably be in its own fsdp unit
# [] test
# [] prune
# [] fix all the code when i have time (maybe jax rewrite?)
# [] add amp
# [] best autowrap policy
# [] add code for running compared to baseline
# [] other losses
# [x] per-process logging


def trivial(rank, world_size):
  with dist_init(rank, world_size):
     torch.cuda.set_device(rank)
     assert rank == torch.cuda.current_device()
     assert rank == dist.get_rank()
     _t = torch.tensor([1, 2, 3], device=rank)
     print("before reduce:", _t)
     dist.all_reduce(_t, op=dist.ReduceOp.SUM)
     print("after reduce:", _t)
     return 0


def main(rank, world_size, group_id, args):
    n_categories = args.n_categories
    n_arm = args.n_arm
    state_dim = args.state_dim
    latent_dim = args.latent_dim
    fc_dim = args.fc_dim
    n_epoch = args.n_epoch
    n_epoch_p = args.n_epoch_p
    min_con = args.min_con
    max_prun_it = args.max_prun_it
    batch_size = args.batch_size
    lam = args.lam
    lam_pc = args.lam_pc
    loss_mode = args.loss_mode
    p_drop = args.p_drop
    s_drop = args.s_drop
    lr = args.lr
    temp = args.temp
    n_run = args.n_run
    device = args.device
    hard = args.hard
    tau = args.tau
    variational = args.variational
    ref_pc = args.ref_pc
    augmentation = args.augmentation
    pretrained_model = args.pretrained_model
    n_pr = args.n_pr
    beta = args.beta
    fsdp = args.fsdp
    min_num_params = args.min_num_params
    cpu_offload = args.offload
    jit_compile = args.jit

    if args.trivial: 
      return trivial(rank, world_size)

    torch.cuda.set_device(rank)
    with dist_init(rank, world_size):
      toml_file = 'pyproject.toml'
      sub_file = 'smartseq_files'
      config = get_paths(toml_file=toml_file, sub_file=sub_file)
      data_path = config['paths']['main_dir'] / config['paths']['data_path']
      data_file = data_path / config[sub_file]['anndata_file']

      folder_name = f'run_{n_run}_K_{n_categories}_Sdim_{state_dim}_aug_{augmentation}_lr_{lr}_n_arm_{n_arm}_nbatch_{batch_size}' + \
                  f'_train.ipynb_nepoch_{n_epoch}_nepochP_{n_epoch_p}'
      saving_folder = config['paths']['main_dir'] / config['paths']['saving_path']
      saving_folder = saving_folder / folder_name
      os.makedirs(saving_folder, exist_ok=True)
      os.makedirs(saving_folder / 'model', exist_ok=True)
      saving_folder = str(saving_folder)

      if augmentation:
          aug_file = config['paths']['main_dir'] / config[sub_file]['aug_model']
      else:
          aug_file = ''
      
      if pretrained_model:
          trained_model = config['paths']['main_dir'] / config[sub_file]['trained_model']
      else:
          trained_model = ''

      # fix torch seed before dataloader

      data = load_data(datafile=data_file)
      trainloader, testloader, _, = get_loaders(dataset=data['log1p'], batch_size=batch_size)

      if args.no_model:
         return 0

      cplMixVAE = cpl_mixVAE(saving_folder=saving_folder, device=rank)
      cplMixVAE.init(categories=n_categories,
                            state_dim=state_dim,
                            input_dim=data['log1p'].shape[1],
                            fc_dim=fc_dim,
                            lowD_dim=latent_dim,
                            x_drop=p_drop,
                            s_drop=s_drop,
                            lr=lr,
                            arms=n_arm,
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
      # -- fsdp -- 

      init_start_event = torch.cuda.Event(enable_timing=True) 
      init_end_event = torch.cuda.Event(enable_timing=True)

      
      model = cplMixVAE.model.to(rank)
      if fsdp:
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params,
        )
        model = FSDP(model, 
                     auto_wrap_policy=my_auto_wrap_policy,
                     cpu_offload=(CPUOffload(offload_params=cpu_offload) if cpu_offload else None),
                     use_orig_params=jit_compile
                     )
      model = torch.compile(model) if jit_compile else model
      opt = torch.optim.Adam(model.parameters(), lr=lr)
      run = None

      wandb.require('core')
      wandb.require('service')
      run = wandb.init(project='dist-mmidas',
                        group=('fsdp-' if fsdp else 'base-') + group_id,
                        config={
                          'n_categories': n_categories,
                          'n_arm': n_arm,
                          'state_dim': state_dim,
                          'latent_dim': latent_dim,
                          'fc_dim': fc_dim,
                          'n_epoch': n_epoch,
                          'n_epoch_p': n_epoch_p,
                          'min_con': min_con,
                          'max_prun_it': max_prun_it,
                          'batch_size': batch_size,
                          'lam': lam,
                          'lam_pc': lam_pc,
                          'loss_mode': loss_mode,
                          'p_drop': p_drop,
                          's_drop': s_drop,
                          'lr': lr,
                          'temp': temp,
                          'n_run': n_run,
                          'hard': hard,
                          'tau': tau,
                          'variational': variational,
                          'ref_pc': ref_pc,
                          'augmentation': augmentation,
                          'n_pr': n_pr,
                          'beta': beta,
                          'fsdp': fsdp,
                          'min_num_params': min_num_params,
                          'cpu_offload': cpu_offload,
                          'rank': rank,
                        },)
                      #  magic=True)
      run.watch(model)

      init_start_event.record()
      if not args.no_train:
        losses = cplMixVAE._fsdp(model,
                          train_loader=trainloader,
                          val_loader=testloader,
                          epochs=n_epoch,
                          n_epoch_p=n_epoch_p,
                          c_p=data['c_p'],
                          c_onehot=data['c_onehot'],
                          min_con=min_con,
                          opt=opt,
                          device = rank,
                          rank=rank,
                          world_size=world_size,
                          max_prun_it=max_prun_it,
                          run=run)
      init_end_event.record()

      # if run is not None:
      #   assert rank == 0
      #   run.finish(quiet=True)

      print(torch.cuda.memory_summary(device=rank, abbreviated=True))
      if rank == 0:
        print(f"Training time: {init_start_event.elapsed_time(init_end_event)}")
        print(f"{model}")
        # print(torch.cuda.memory_snapshot())


    # model_file = cplMixVAE.train(train_loader=trainloader,
    #                             test_loader=testloader,
    #                             n_epoch=n_epoch,
    #                             n_epoch_p=n_epoch_p,
    #                             c_onehot=data['c_onehot'],
    #                             c_p=data['c_p'],
    #                             min_con=min_con,
    #                             max_prun_it=max_prun_it)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_categories", default=120, type=int, help="(maximum) number of cell types")
  parser.add_argument("--state_dim", default=2, type=int, help="state variable dimension")
  parser.add_argument("--n_arm", default=2, type=int,  help="number of mixVAE arms for each modalities")
  parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
  parser.add_argument("--tau",  default=.005, type=float, help="softmax temperature")
  parser.add_argument("--beta",  default=.01, type=float, help="KL regularization parameter")
  parser.add_argument("--lam",  default=1, type=float, help="coupling factor")
  parser.add_argument("--lam_pc",  default=1, type=float, help="coupling factor for ref arm")
  parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
  parser.add_argument("--n_epoch", default=25, type=int, help="Number of epochs to train")
  parser.add_argument("--n_epoch_p", default=10000, type=int, help="Number of epochs to train pruning algorithm")
  parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
  parser.add_argument("--max_prun_it", default=50, type=int, help="maximum number of pruning iteration")
  parser.add_argument("--ref_pc", default=False, type=bool, help="path of the data augmenter")
  parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
  parser.add_argument("--batch_size", default=5000, type=int, help="batch size")
  parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
  parser.add_argument("--augmentation", default=False, type=bool, help="enable VAE-GAN augmentation")
  parser.add_argument("--lr", default=.001, type=float, help="learning rate")
  parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
  parser.add_argument("--s_drop", default=0.2, type=float, help="state probability of dropout")
  parser.add_argument("--pretrained_model", default=False, type=bool, help="use pretrained model")
  parser.add_argument("--n_pr", default=0, type=int, help="number of pruned categories in case of using a pretrained model")
  parser.add_argument("--loss_mode", default='MSE', type=str, help="loss mode, MSE or ZINB")
  parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
  parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
  parser.add_argument("--device", default='cuda', type=str, help="computing device, either 'cpu' or 'cuda'.")
  # test this
  parser.add_argument("--fsdp", action='store_true', help="enable distributed training")
  parser.add_argument("--min_num_params", default=200, type=int, help="minimum number of parameters for autowrap policy in distributed training")
  parser.add_argument("--offload", action='store_true', help="enable cpu offload in distributed training")
  parser.add_argument("--jit", action='store_true', help="enable jit compile in distributed training")
  parser.add_argument("--no_train", action='store_true', help="disable training")
  parser.add_argument("--trivial", action='store_true', help="run trivial example")
  parser.add_argument("--no_model", action='store_true', help="disable model creation")

  args = parser.parse_args()
  WORLD_SIZE = torch.cuda.device_count()
  # main(0, WORLD_SIZE, args)
  mp.spawn(main, args=(WORLD_SIZE, wandb.util.random_string(4), args), nprocs=WORLD_SIZE, join=True)

# tgtft271