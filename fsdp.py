import argparse
import functools
import os
import random
import string
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import timedelta
from functools import reduce
from itertools import filterfalse

# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install pyrsistent wandb tqdm matplotlib
# python fsdp.py --dataset mnist --model net --parallel none --epochs 5
# python fsdp.py --dataset mnist --model net --world-size 2 --no-sampler --epochs 2

# import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, CPUOffload)
from torch.distributed.fsdp.wrap import (always_wrap_policy, enable_wrap,
                                         size_based_auto_wrap_policy, wrap)
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.overrides as overrides
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from pyrsistent import PMap, PVector
PMap.__call__ = lambda self, x: self[x]
PVector.__call__ = lambda self, x: self[x]
from pyrsistent import pmap
from pyrsistent import pvector
from pyrsistent import m
from pyrsistent import v
import wandb
from tqdm import trange

import mmidas
from my_utils import conj, try_to_get, pprint, avg, dprint, make_path, random_string, convert, random_of
from torch_utils import print_available_torch, ResourceLogger, count


def take_percent_data(dataset, percent):
  return torch.utils.data.Subset(dataset, list(range(int(len(dataset) * percent))))

def check_not_a100(rank, backend):
  assert not ("A100" in torch.cuda.get_device_name(rank) and backend == 'nccl')

def record_memory_history(fname, folder='memory-snapshots'):
  make_path(folder)
  fname = f"{folder}/{fname}"
  torch.cuda.memory._dump_snapshot(fname)
  dprint(f"> saved memory snapshot to {fname}")

    
def get_device_name_or_empty(rank):
  return torch.cuda.get_device_name(rank) if rank == 'cuda' \
         or isinstance(rank, int) else ''

def make_wandb(project, dataset, id, config):
  wandb.require('service')
  return wandb.init(project=project, group=f"{dataset}-{id}", config=dict(config))

def make_run_config(parser):
  config = pmap(filter_none(vars(parser.parse_args()).items()))
  config = try_to_get(config, 'id', random_of('str', 4))
  return config

def is_none(*xs):
  match xs:
    case (v,):
      return v is None or v == '' or v == 'None' or v == 'none' or v == False
    case (_, v):
      return v is None or v == '' or v == 'None' or v == 'none' or v == False

def filter_none(xs):
  return filterfalse(lambda kv: is_none(*kv), xs)

@contextmanager
def profile_run(rank, id):
  try:
    with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
      yield prof
  finally:
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
    make_path('traces')
    prof.export_chrome_trace(f"trace-{time.time()}-{id}-{rank}.json")


# TODO
def print_summary(epoch_avg, mem_avg, all_mem_avg, model):
  dprint(" -- summary -- ", bold=True)

class DeepestNet(nn.Module):
  def __init__(self):
    super(DeepestNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 9000)
    self.fc1a = nn.Linear(9000, 1000)
    self.fc = nn.ModuleList([nn.Linear(1000, 1000) for _ in range(176)])
    self.fc2a = nn.Linear(1000, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc1a(x)
    for l in self.fc:
      x = l(x)
    x = self.fc2a(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output
  
class DeepNet(nn.Module):
  def __init__(self):
    super(DeepNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 9000)
    self.fc1a = nn.Linear(9000, 1000)
    self.fc1b = nn.Linear(1000, 1000)
    self.fc1c = nn.Linear(1000, 1000)
    self.fc1d = nn.Linear(1000, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc1a(x)
    x = self.fc1b(x)
    x = self.fc1c(x)
    x = self.fc1d(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

# TODO: make this work on general functions
def plot(data, plot_t, xlabel, ylabel, title, legend, fname, folder=''):
  if plot_t == 'line':
    plotter = plt.plot
  elif plot_t == 'scatter':
    plotter = plt.scatter
  else:
    raise ValueError(f"invalid plot type: {plot_t}")
  
  plotter(range(len(data)), data)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.legend([legend])
  if not os.path.exists(folder) and not (folder == ''): 
    os.makedirs(folder)
  plt.savefig(f"{folder}/{fname}")
  dprint(f"> saved plot to {folder}/{fname}")
  plt.close()


def setup_torch_distributed_(rank, world_size, backend='nccl', timeout=120):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dprint(f"rank {rank} - master addr: {os.environ['MASTER_ADDR']}")
  dprint(f"rank {rank} - master port: {os.environ['MASTER_PORT']}")
  # check_not_a100(rank, backend)
  if backend == 'gloo':
    dprint(f"warning: using gloo backend")
  dist.init_process_group(backend=backend, rank=rank, world_size=world_size, 
                          timeout=timedelta(seconds=timeout))

def cleanup_torch_distributed_():
  dist.destroy_process_group()

def make_cuda_str(rank):
  if type(rank) == int:
    return f"cuda:{rank}"
  return rank

def make_dataset_mnist():
  transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ])
  dataset1 = datasets.MNIST('../data', train=True, download=False, 
                            transform=transform)
  dataset2 = datasets.MNIST('../data', train=False, transform=transform)
  return dataset1, dataset2

def make_dataset(name):
  match name:
    case 'mnist':
      return make_dataset_mnist()

def make_loaders(dataset, parallel, batch_size, test_batch_size, percent, **kwargs):
  match dataset, parallel:
    case 'mnist', True:
      dprint(f"> making dataloader: mnist (parallel: {parallel})")
      rank = kwargs['rank']
      world_size = kwargs['world_size']

      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ])
      dataset1 = datasets.MNIST('../data', train=True, download=False, 
                                transform=transform)
      dataset2 = datasets.MNIST('../data', train=False, transform=transform)
      dataset1 = take_percent_data(dataset1, percent)
      dataset2 = take_percent_data(dataset2, percent)
      sampler1 = DistributedSampler(dataset1, rank=rank, 
                                    num_replicas=world_size, shuffle=True)
      sampler2 = DistributedSampler(dataset2, rank=rank, 
                                    num_replicas=world_size)
      train_kwargs = {'batch_size': batch_size, 'sampler': sampler1}
      test_kwargs = {'batch_size': test_batch_size, 'sampler': sampler2}
      cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False, 
                     'drop_last': True}
      train_kwargs.update(cuda_kwargs)
      test_kwargs.update(cuda_kwargs)
      train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
      test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
      return train_loader, test_loader, sampler1
    case 'mnist', False:
      dprint(f"> making dataloader: mnist (parallel: {parallel})")
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ])
      dataset1 = datasets.MNIST('../data', train=True, download=False, transform=transform)
      dataset2 = datasets.MNIST('../data', train=False, transform=transform)
      dataset1 = take_percent_data(dataset1, percent)
      dataset2 = take_percent_data(dataset2, percent)
      train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True, drop_last=True)
      test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size, shuffle=False, drop_last=True)
      return train_loader, test_loader, None
    case 'mmidas', True:
      raise NotImplementedError # TODO
    case 'mmidas', False:

      raise NotImplementedError # TODO
    
def mmidas_data():
  ...

def mmidas_dataloaders(batch_size):
  ...
  
def make_model(name):
  model = None
  match name:
    case 'net':
      model = Net()
    case 'deep':
      model = DeepNet()
    case 'deepest':
      model = DeepestNet()
    case 'mmidas':
      raise NotImplementedError # TODO
    case _:
      raise ValueError(f"invalid model: {name}")
  return model

def optimize_model(model, parallel, is_jit, rank, wrap, min_params=20000, offload=None):
  if parallel == 'fsdp':
    my_auto_wrap_policy = None
    match wrap:
      case 'size_based':
        my_auto_wrap_policy = functools.partial(
          size_based_auto_wrap_policy, min_num_params=min_params
        )
      case 'always':
        my_auto_wrap_policy = always_wrap_policy
      case 'none':
        my_auto_wrap_policy = None
      case _:
        raise ValueError(f"invalid wrap policy: {wrap}")
    cpu_offload = CPUOffload(offload) if offload else None
    # maybe pass in device_id
    assert rank == torch.cuda.current_device()
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, 
                 cpu_offload=cpu_offload, use_orig_params=is_jit, 
                 device_id=rank)
  elif parallel == 'ddp':
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
  if is_jit:
    model = torch.compile(model)
  return model


def train(model, rank, world_size, train_loader, optimizer, epoch, sampler=None, parallel=True, run=None, print_loss=True):
  model.train()
  ddp_loss = torch.zeros(2).to(rank)
  mem = []
  if sampler:
    sampler.set_epoch(epoch)
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(rank), target.to(rank) # TODO
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target, reduction='sum')
    _mem_alloc = convert(torch.cuda.memory_allocated(rank), 'B', 'MB')
    mem.append(_mem_alloc)
    if run:
      run.log({f'cuda {rank} memory allocated': _mem_alloc})
    loss.backward()
    optimizer.step()
    ddp_loss[0] += loss.item()
    ddp_loss[1] += len(data)
  if parallel:
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
  if print_loss:
    dprint('Train Epoch: {} \tLoss: {:.6f}, \t Rank {} memory allocated: {}'.format(
      epoch, ddp_loss[0] / ddp_loss[1], rank, mem[-1]))
  if run:
    assert torch.cuda.max_memory_allocated() \
           == torch.cuda.max_memory_allocated(rank)
    run.log({'train_loss': ddp_loss[0] / ddp_loss[1],
             f'rank {rank} max memalloc': \
                 convert(torch.cuda.max_memory_allocated(rank), 'B', 'MB'),})
  return mem

def test(model, rank, world_size, test_loader, parallel=True, print_loss=True):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
      batches = 0
      for data, target in test_loader:
        data, target = data.to(rank), target.to(rank)
        output = model(data)
        ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
        ddp_loss[2] += len(data)
        batches += 1
    if parallel:
      dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if print_loss:
      test_loss = ddp_loss[0] / ddp_loss[2]
      dprint('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
          100. * ddp_loss[1] / ddp_loss[2]))

def start_run(rank, world_size, run, config):
  parallel = 'parallel' in config
  is_fsdp = 'parallel' in config and config.parallel == 'fsdp'
  master_rank = rank == 0 or not parallel

  _loaders = make_loaders(dataset=config.dataset, 
                               parallel=(parallel and 'sampler' in config),
                               batch_size=config.batch_size,
                               test_batch_size=config.test_batch_size, rank=rank,
                               world_size=world_size, percent=config.percent)
                               
  train_loader, test_loader, sampler = _loaders
  model = make_model(config.model)
  model = model.to(rank) if not is_fsdp else model
  if parallel:
    model = optimize_model(model, config.parallel, 'jit' in config, rank,
                            config.wrap, config.min_params, 
                            'cpu_offload' in config)

  optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
  
  rets = {
    'model': model,
    'epoch_times': [],
    'mems': [],
  }
  pbar = (trange(config.epochs, colour='red') if master_rank
          else range(config.epochs))
  for epoch in pbar:
    start = time.time()
    rets['mems'] += train(model, rank, world_size, train_loader, 
                          optimizer, epoch, sampler=sampler, parallel=parallel,
                          run=run, print_loss=master_rank)
    test(model=model, rank=rank, world_size=world_size, 
         test_loader=test_loader, parallel=parallel, 
         print_loss=master_rank)
    scheduler.step()
    rets['epoch_times'].append(time.time() - start)
    if run:
      run.log({'seconds per epoch': rets['epoch_times'][-1]})
  return rets

def main(rank, world_size, config):
  parallel = 'parallel' in config
  master_rank = rank == 0 or not parallel
  run = None
  mlogger = None
  init_start_event = torch.cuda.Event(enable_timing=True)
  init_end_event = torch.cuda.Event(enable_timing=True)
  epoch_times = []
  mems = []
  rets = {}
  mems_tensor = torch.zeros(world_size, device=rank)
  _name = get_device_name_or_empty(rank)

  if parallel:
    assert torch.cuda.is_available()
    setup_torch_distributed_(rank, world_size, config.backend, config.timeout)
    torch.cuda.set_device(rank)
    
  dprint(f"> training on {make_cuda_str(rank)} {_name} \
           (host: {os.uname().nodename})")

  if 'wandb' in config: 
    run = make_wandb('dist-mmidas', config.dataset, config.id, config)

  if 'plot' in config and 'memory' in config.plot:
    mlogger = ResourceLogger(world_size, interval=config.interval, run=run,
                            rank=rank)
    
  init_start_event.record()
  if 'profile' in config and master_rank:
    with profile_run(rank, config.id) as prof:
      rets = start_run(rank, world_size, run, config)
  else:
    rets = start_run(rank, world_size, run, config)
  init_end_event.record()
  epoch_times += rets['epoch_times']
  mems += rets['mems']
  model = rets['model']

    # if args.record_memory_history:
    #   torch.cuda.memory._record_memory_history()

    # if args.log_after_epoch < 0 and mlogger is not None:
    #   assert not mlogger.running
    #   mlogger.start()

    #   if args.log_after_epoch == epoch and mlogger is not None and i == 0:
    #     assert not mlogger.running
    #     mlogger.start()
  
  if mlogger:
    mlogger.stop()
  
  if parallel:
    mems_tensor[rank] = avg(mems)
    dist.all_reduce(mems_tensor, op=dist.ReduceOp.SUM)

  if master_rank:
    dprint(" -- summary -- ", bold=True)

  if parallel:
    if run:
      run.log({'avg seconds per epoch': avg(epoch_times),
              f'cuda {rank} average memory allocated': mems_tensor[rank],
               'avg memory allocated across all gpus': mems_tensor.mean()})
    if master_rank:
      dprint(f"avg seconds per epoch: {avg(epoch_times)}")
      dprint(f"cuda {rank} average memory allocated: {mems_tensor[rank]}")
      dprint(f"avg memory allocated across all gpus: {mems_tensor.mean()}")
  else:
    if run: 
      run.log({'avg seconds per epoch': avg(epoch_times),
              f'cuda {torch.cuda.current_device()} average memory allocated': avg(mems),
              'avg memory allocated across all gpus': avg(mems)})
    if master_rank:
      dprint(f"avg seconds per epoch: {avg(epoch_times)}")
      dprint(f"cuda {torch.cuda.current_device()} average memory allocated: \
               {avg(mems)}")
      dprint(f"avg memory allocated across all gpus: {avg(mems)}")
      
  if 'record_memory_history' in config:
    record_memory_history(f"{time.time()}-{config.id}-{rank}.pickle")

  if rank == 0 or rank == 'cuda':
    torch.cuda.synchronize()
    dprint(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")

  if master_rank:
    dprint(f"{model}")
    dprint(f"number of parameters: {count(model, 'params')}")

  if 'save_model' in config:
    if parallel:
      dist.barrier() # TODO: might give bugs
    if master_rank:
      torch.save(model.state_dict(), "mnist_cnn.pt")

  if parallel:
    cleanup_torch_distributed_()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=546, metavar='S',
                        help='random seed (default: 546)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--nccl-debug', type=str, default=None, 
                        help='NCCL debug level (default: None)')
    parser.add_argument('--backend', type=str, default='nccl', 
                        help='distributed backend (default: nccl)')
    parser.add_argument('--multinode', action='store_true', default=False, 
                        help='enable multinode training') # TODO
    parser.add_argument('--min-params', type=int, default=20000, 
                        help='minimum number of parameters to wrap')
    parser.add_argument('--cpu-offload', action='store_true', default=False, 
                        help='enable CPU offload')
    parser.add_argument('--timeout', type=int, default=120, 
                        help='timeout for distributed ops (default: 120)')
    parser.add_argument('--implementation', type=str, default='torch', 
                        help='implementation, torch or jax (default: torch)') # TODO
    parser.add_argument('--mixed-precision', action='store_true', 
                        default=False, 
                        help='use mixed precision for faster training') # TODO
    parser.add_argument('--jit', action='store_true', default=False, 
                        help='jit compile the model for faster training') # TODO
    parser.add_argument('--wandb', action='store_true', default=False, 
                        help='log to wandb') # TODO
    parser.add_argument('--device', type=str, default='cuda', 
                        help='device to use (default: cuda)')
    parser.add_argument('--dataset', type=str, default='mmidas', 
                        help='name of the dataset to train on (default: mnist). \
                        Options: smartseq, 10x, mnist') # TODO
    parser.add_argument('--no_fsdp', action='store_true', default=False, 
                        help='disable fsdp')
    parser.add_argument('--world-size', type=int, default=None,
                         help='world size for distributed training \
                               (default: None)')
    parser.add_argument('--no-sampler', action='store_true', default=False, 
                        help='disable distributed sampler')
    parser.add_argument('--backward_prefetch', action='store_true', 
                        default=False, help='enable backward prefetching') # TODO
    parser.add_argument('--wrap', type=str, default='size_based', 
                        choices=['size_based', 'always', 'none'], 
                        help='fsdp wrap policy (default: size_based). \
                          Options: size_based, always, none') # TODO
    parser.add_argument('--plot', nargs='+', default=['time', 'loss'], 
                        help='plot memory usage (default: memory)') # TODO
    parser.add_argument('--plot-style', type=str, choices=['line', 'scatter'], 
                        default='line', help='plot type (default: line)')
    parser.add_argument('--plot-backend', type=str, default='matplotlib', 
                        help='plot backend (default: wandb). \
                              Options: matplotlib, plotly, seaborn') # TODO
    parser.add_argument('--model', type=str, default='net', 
                        help='model to train (default: net). \
                              Options: mmidas, net') # TODO
    parser.add_argument('--parallel', type=str, default='fsdp', 
                        help='parallel training method (default: fsdp). \
                              Options: fsdp, ddp, none') # TODO
    parser.add_argument('--no-loss', nargs='+', default=[], 
                        help='losses to disable (default: [])')
    parser.add_argument('--record-memory-history', action='store_true', 
                        default=False, help='record memory history')
    parser.add_argument('--tensor-parallel', action='store_true', 
                        default=False, help='use tensor parallelism') # TODO
    parser.add_argument('--interval', type=float, default=1, 
                        help='memory logging interval (default: 0.005)')
    parser.add_argument('--log-after-epoch', type=int, default=-1, 
                        help='log after epoch (default: -1)') # TODO
    parser.add_argument('--runs', type=int, default=1, 
                        help='number of runs (default: 1)') # TODO
    parser.add_argument('--percent', type=float, default=1.0, 
                        help='percent of data to use (default: 1.0)') # TODO
    parser.add_argument('--id', type=str, default='', 
                        help='experiment id (default: random)') # TODO
    parser.add_argument('--profile', action='store_true', default=False, 
                        help='enable torch profiler') # TODO
    parser.add_argument('--sampler', type=str, default='distributed',
                        help='sampler to use (default: distributed)') # TODO
    # mmidas-smartseq args
    # parser.add_argument('--categories', type=int, default=120, help="(maximum) number of cell types (default: 120)")
    # parser.add_argument('--state_dim', type=int, default=2, help="state variable dimension (default: 2)")
    # parser.add_argument('--arms', type=int, default=2, help="number of mixVAE arms for each modality (default: 2)")
    # parser.add_argument('--temp', type=float, default=1.0, help="gumbel-softmax temperature (default: 1.0)")
    # parser.add_argument('--tau', type=float, default=.005, help="softmax temperature (default: .005)")
    # parser.add_argument('--beta', type=float, default=.01, help="KL regularization parameter (default: .01)")
    # parser.add_argument('--lam', type=float, default=1.0, help="coupling factor (default: 1.0)")
    # parser.add_argument('--lam_pc', type=float, default=1.0, help="coupling factor for ref arm (default: 1.0)")
    # parser.add_argument('--latent_dim', type=int, default=10, help="latent dimension (default: 10)")
    # parser.add_argument('--epochs2', type=int, default=10000, help="Number of epochs to train (default: 10000)")
    # parser.add_argument('--epochs_p', type=int, default=10000, help="Number of epochs to train pruning algorithm (default: 10000)")
    # parser.add_argument('--min_con', type=float, default=.99, help="minimum consensus (default: .99)")
    # parser.add_argument('--max_prun_it', type=int, default=50, help="maximum number of pruning iterations (default: 50)")
    # parser.add_argument('--ref_pc', action='store_true', default=False, help="use a reference prior component")
    # parser.add_argument('--fc_dim', type=int, default=100, help="number of nodes at the hidden layers (default: 100)")
    # parser.add_argument('--batch_size2', type=int, default=5000, help="batch size (default: 5000)")
    # parser.add_argument('--no_variational', action='store_true', default=False, help="enable variational mode")
    # parser.add_argument('--augmentation', action='store_true', default=False, help="enable VAE-GAN augmentation")
    # parser.add_argument('--lr2', type=float, default=.001, help="learning rate (default: .001)")
    # parser.add_argument('--p_drop', type=float, default=0.5, help="input probability of dropout (default: 0.5)")
    # parser.add_argument('--s_drop', type=float, default=0.2, help="state probability of dropout (default: 0.2)")
    # parser.add_argument('--pretrained', action='store_true', default=False, help="use pretrained model")
    # parser.add_argument('--n_pr', type=int, default=0, help="number of pruned categories in case of using a pretrained model (default: 0)")
    # parser.add_argument('--loss_mode', type=str, default='MSE', help="loss mode, MSE or ZINB (default: MSE)")
    # parser.add_argument('--runs', type=int, default=1, help="number of the experiment (default: 1)")
    # parser.add_argument('--hard', action='store_true', default=False, help="hard encoding")
    config = make_run_config(parser)
    torch.manual_seed(config.seed)
    if 'nccl_debug' in config:
      os.environ['NCCL_DEBUG'] = config.nccl_debug.upper()

    dprint("config:", color='red', bold=True)
    pprint(config)
    dprint()
    dprint(f"visible cuda devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
    if 'parallel' in config:
      world_size = config.get('world_size', torch.cuda.device_count())
      dprint(f"world size: {world_size}")
      mp.spawn(main, args=(world_size, config), nprocs=world_size, join=True)
    else:
      world_size = config.get('world_size', 1)
      rank = config.device
      dprint(f"world size: {world_size}")
      dprint(f"rank: {rank}")
      main(rank, world_size, config)

# TODO
# [] --plot-gpu flag
# [] --plot-time flag
# [] --plot-backend flag
# [] --multinode w/ srun
# [] mmidas-smartseq no parallel

# [] handle mmidas vs mnist repeated args
# [] multiple runs flag?
# [] allow people to run without wandb
# [] print out flags to console
# [] maybe decouple dataset from dataloader
# [] test no sampler w/o printing
# [] fix mnist main
# [] play with fsdp settings
# [] options for memory logging format
# [] make --plot take a list of params
# [] allow you to change the plot memory interval
# [] give different seeds to different processes


# if time:
# [] test w/ multiple nodes

# remember: srun vs sbatch vs torchrun
# [] remove seconds at beginning from plot

# [] fix plot legend
# [] add different default run configs
# [] change the way I log memory
# [] try out pytorch ignite
# [] add config to wandb log
# [] setup_torch_distributed_ github utils
# [] default cuda device of new thread is cuda:0

# when num params is ~1mil, you don't see that much impact from

# 5 epochs, 2-3 times each
# if time, try ddp
# try ddp with the constructor
# results with
# (i) shallow network with ~1 million
# (ii) deep network with ~93 million params


# mmidas 
# n_arms = 2 -> ~22 million
# n_arms = 5 -> ~50 million
# n_arms = 10 -> ~100 million

# DeepNet: see if I can get to ~50 million params
# multiple of 8 tensor sizes
