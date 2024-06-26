import os
import argparse
import functools
import torch
import torch.nn as nn 

import torch.overrides as overrides
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms 

from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist 
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
  CPUOffload,
  BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
  size_based_auto_wrap_policy,
  enable_wrap,
  wrap,
)

from datetime import timedelta

from tqdm import trange

import mmidas

import wandb

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

def trivial(rank, world_size, args):
  setup(rank, world_size)
  torch.cuda.set_device(rank)
  assert torch.cuda.current_device() == rank
  assert dist.get_rank() == rank
  t = torch.tensor([1, 2, 3], device=rank)
  print(f"rank {rank} - before reduce: {t}")
  dist.all_reduce(t, op=dist.ReduceOp.SUM)
  print(f"rank {rank} - after reduce: {t}")
  cleanup()
  return 0

def dprint(*args, **kwargs):
  if __debug__:
    print(*args, **kwargs)

def print_gpus():
  dprint(f"gpus: {torch.cuda.device_count()}")
  for i in range(torch.cuda.device_count()):
    dprint(f"\tgpu {i}: {torch.cuda.get_device_name(i)}")

def eparams(model):
  return list(model.parameters())

def setup(rank, world_size, backend='nccl', is_multinode=False, timeout=120, parallel=True):
  if not parallel:
    return
  if is_multinode:
    os.environ['MASTER_ADDR'] = os.environ['SLURM_SUBMIT_HOST']
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
  else:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
  if rank == 0:
    dprint(f"master addr: {os.environ['MASTER_ADDR']}")
    dprint(f"master port: {os.environ['MASTER_PORT']}")
  assert not ("A100" in torch.cuda.get_device_name(rank) and backend == 'nccl'), "a100 doesn't support nccl"
  dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=timeout))

def cleanup(parallel=True):
  if not parallel:
    return
  dist.destroy_process_group()

def rank2dev(rank):
  if type(rank) == int:
    return f"cuda:{rank}"
  return rank

def make_data_loaders(task, parallel, batch_size, test_batch_size, **kwargs):
  match task, parallel:
    case 'mnist', True:
      rank = kwargs['rank']
      world_size = kwargs['world_size']

      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ])
      dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
      dataset2 = datasets.MNIST('../data', train=False, transform=transform)
      sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
      sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)
      train_kwargs = {'batch_size': batch_size, 'sampler': sampler1}
      test_kwargs = {'batch_size': test_batch_size, 'sampler': sampler2}
      cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
      train_kwargs.update(cuda_kwargs)
      test_kwargs.update(cuda_kwargs)
      train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
      test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
      return train_loader, test_loader, sampler1
    case 'mnist', False:
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))
      ])
      dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
      dataset2 = datasets.MNIST('../data', train=False, transform=transform)
      train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
      test_loader = torch.utils.data.DataLoader(dataset2, batch_size=test_batch_size, shuffle=False)
      return train_loader, test_loader, None
    case 'mmidas', True:
      raise NotImplementedError # TODO
    case 'mmidas', False:
      raise NotImplementedError # TODO
  
def make_model(name, parallel, rank=None):
  model = None
  dest = None
  if rank == 0 or not parallel:
    dprint(f"making model: {name}")
  match name:
    case 'mnist':
      model = Net()
    case 'mmidas':
      raise NotImplementedError # TODO
  return model.to(rank)

def transform_model(model, parallel, is_fsdp, rank, no_wrap=False, min_params=1000, offload=None):
  if not parallel:
    return model
  if is_fsdp:
    if rank == 0:
      dprint(f"> wrapping model with fsdp")
      dprint(f"min params: {min_params}")
      dprint(f"no wrap: {no_wrap}")
      dprint(f"cpu offload: {offload}")
    my_auto_wrap_policy = functools.partial(
      size_based_auto_wrap_policy, min_num_params=min_params
    ) if not no_wrap else None
    cpu_offload = CPUOffload(offload) if offload else None
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=cpu_offload)
  return model


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None, parallel=True):
  model.train()
  ddp_loss = torch.zeros(2).to(rank)
  if sampler:
    sampler.set_epoch(epoch)
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(rank), target.to(rank)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target, reduction='sum')
    loss.backward()
    optimizer.step()
    ddp_loss[0] += loss.item()
    ddp_loss[1] += len(data)
  if parallel:
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
  if rank == 0 or not parallel:
    dprint('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

def test(model, rank, world_size, test_loader, parallel=True):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)
    if parallel:
      dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0 or not parallel:
        test_loss = ddp_loss[0] / ddp_loss[2]
        dprint('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))

def fsdp_main(rank, world_size, args):
  parallel = not args.no_parallel
  is_fsdp = not args.no_fsdp

  if parallel:
    assert torch.cuda.is_available()

  _name = f"({torch.cuda.get_device_name(rank)})" if rank == 'cuda' or isinstance(rank, int) else ''
  dprint(f"> training on {rank2dev(rank)} {_name}")

  setup(rank, world_size, args.backend, args.multinode, args.timeout, parallel=parallel)

  train_loader, test_loader, sampler = make_data_loaders(task=args.task, parallel=parallel,
                                                batch_size=args.batch_size,
                                                test_batch_size=args.test_batch_size, rank=rank, world_size=world_size)
  if parallel:
    local_rank = rank if not args.multinode else int(os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(local_rank)

  # if args.time_cuda:
  init_start_event = torch.cuda.Event(enable_timing=True)
  init_end_event = torch.cuda.Event(enable_timing=True)

  model = make_model(args.task, parallel, rank=rank)
  model = transform_model(model, parallel, is_fsdp, rank, args.no_wrap, args.min_params, args.cpu_offload)
  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  # if args.time_cuda:
  init_start_event.record()
  if not args.no_train:
    for epoch in trange(1, args.epochs + 1):
      train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler, parallel=parallel)
      test(model, rank, world_size, test_loader, parallel=parallel)
      scheduler.step()
  # if args.time_cuda:
  init_end_event.record()

  # if args.time_cuda:
  if rank == 0 or rank == 'cuda':
    print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
  if rank == 0 or not parallel: 
    print(f"{model}")

  # save(save_model=args.save_model, parallel=parallel, rank=rank, model=args.)
  if args.save_model:
     if parallel:
      dist.barrier()
     states = model.state_dict()
     if rank == 0 or not parallel:
         torch.save(states, "mnist_cnn.pt")
  cleanup(parallel=parallel)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--nproc_per_node', type=int, default=4, metavar='N', help='number of processes per node (default: 4)')
    parser.add_argument('--nccl_debug', action='store_true', default=False, help='enable NCCL debugging')
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend (default: nccl)')
    parser.add_argument('--multinode', action='store_true', default=False, help='enable multinode training') # TODO
    parser.add_argument('--no_parallel', action='store_true', default=False, help='disable parallelism')
    parser.add_argument('--min_params', type=int, default=1000, help='minimum number of parameters to wrap')
    parser.add_argument('--no_wrap', action='store_true', default=False, help='disable wrapping')
    parser.add_argument('--cpu_offload', action='store_true', default=False, help='enable CPU offload')
    parser.add_argument('--timeout', type=int, default=120, help='timeout for distributed ops (default: 120)')
    parser.add_argument('--no_train', action='store_true', default=False, help='disable training') # TODO
    parser.add_argument('--trivial', action='store_true', default=False, help='run trivial test')
    parser.add_argument('--jax', action='store_true', default=False, help='use jax implementation') # TODO
    parser.add_argument('--torch', action='store_true', default=False, help='use torch implementation') # TODO
    parser.add_argument('--mixed', action='store_true', default=False, help='use mixed precision for faster training') # TODO
    parser.add_argument('--jit', action='store_true', default=False, help='jit compile the model for faster training') # TODO
    parser.add_argument('--wandb', action='store_true', default=False, help='log to wandb') # TODO
    parser.add_argument('--device', type=str, default='cuda', help='device to use (default: cuda)')
    parser.add_argument('--task', type=str, default='mmidas', help='name of the model to train (default: mmidas). Options: mmidas, mnist, trivial') # TODO
    parser.add_argument('--no_fsdp', action='store_true', default=False, help='disable fsdp')
    parser.add_argument('--time_cuda', action='store_true', default=False, help='time cuda ops')
    args = parser.parse_args()

    if args.wandb:
      args.id = wandb.util.random_string(4)
      dprint(f"wandb id: {args.id}")

    assert not (args.no_parallel and args.multinode), "cannot disable parallelism and enable multinode training"
    assert not (not args.no_parallel and (args.device == 'cpu' or args.device == 'mps')), "cannot disable parallelism and use cpu or mps"

    torch.manual_seed(args.seed)

    if args.nccl_debug:
      os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker'
    # os.environ['NCCL_IB_DISABLE'] = '0'
    # os.environ['NCCL_IB_HCA'] = 'mlx5_1'
    
    dprint(f"backend: {args.backend}")
    dprint(f"parallel training: {not args.no_parallel}")
    dprint(f"multinode training: {args.multinode}")
    dprint(f"task: {args.task}")
    dprint(f"cpu available: {torch.cpu.is_available()}")
    dprint(f"cuda available: {torch.cuda.is_available()}")
    print_gpus()
    dprint(f"mps available: {torch.backends.mps.is_available()}")
    if args.trivial:
      world_size = torch.cuda.device_count()
      dprint(f"world size: {world_size}")
      mp.spawn(trivial,
               args=(world_size, args),
               nprocs=world_size,
               join=True)
    elif args.no_parallel:
      world_size = 1
      rank = args.device
      dprint(f"world size: {world_size}")
      dprint(f"rank: {rank}")
      fsdp_main(rank, world_size, args)
      # main(args)
    elif args.multinode:
      assert torch.cuda.device_count() == int(os.environ['SLURM_GPUS_ON_NODE'])
      world_size = int(os.environ['SLURM_NTASKS'])
      rank = int(os.environ['SLURM_PROCID'])
      local_rank = int(os.environ['SLURM_LOCALID'])
      dprint(f"world size: {world_size}")
      dprint(f"rank: {rank}")
      dprint(f"local rank: {local_rank}")
      fsdp_main(rank, world_size, args)
    else:
      world_size = torch.cuda.device_count()
      dprint(f"world size: {world_size}")
      mp.spawn(fsdp_main,
          args=(world_size, args),
          nprocs=world_size,
          join=True)
      

  

# [] get slurm world size
# [] test cuda device count on sbatch job
