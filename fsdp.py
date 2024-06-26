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
  dprint(f"cuda available: {torch.cuda.is_available()}")
  dprint(f"gpus: {torch.cuda.device_count()}")
  for i in range(torch.cuda.device_count()):
    dprint(f"\tgpu {i}: {torch.cuda.get_device_name(i)}")

def eparams(model):
  return list(model.parameters())

def setup(rank, world_size, backend='nccl', is_multinode=False, timeout=120):
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

def cleanup():
  dist.destroy_process_group()

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
        
def main(args):
  assert args.no_parallel

  dprint(f"> training on {args.device}")
  if args.device == 'cuda':
    dprint(f"> device name: {torch.cuda.get_device_name()}")

  transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])
   
  dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
  dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, num_workers=0)
  test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
  model = Net().to(args.device)
  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  for epoch in trange(1, args.epochs + 1):
     train(args, model, args.device, -1, train_loader, optimizer, epoch, sampler=None, parallel=not args.no_parallel)
     test(model, args.device, -1, test_loader, parallel=not args.no_parallel)
     scheduler.step()
  print(f"{model}")
  if args.save_model:
     states = model.state_dict()
     if rank == 0:
         torch.save(states, "mnist_cnn.pt")

def fsdp_main(rank, world_size, args):
  assert not args.no_parallel

  if args.trivial:
    return trivial(rank, world_size, args)

  setup(rank, world_size, args.backend, args.multinode, args.timeout)

  transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])
   
  dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
  dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
  sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
  sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

  train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
  test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
  cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
  train_kwargs.update(cuda_kwargs)
  test_kwargs.update(cuda_kwargs)

  train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

  local_rank = rank if not args.multinode else int(os.environ['SLURM_LOCALID'])
  torch.cuda.set_device(local_rank)

  init_start_event = torch.cuda.Event(enable_timing=True)
  init_end_event = torch.cuda.Event(enable_timing=True)

  model = Net().to(rank)

  my_auto_wrap_policy = functools.partial(
     size_based_auto_wrap_policy, min_num_params=args.min_params
  ) if not args.no_wrap else None
  offload = CPUOffload(args.cpu_offload) if args.cpu_offload else None
  model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=offload)

  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  init_start_event.record()
  if not args.no_train:
    for epoch in trange(1, args.epochs + 1):
      train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1, parallel=not args.no_parallel)
      test(model, rank, world_size, test_loader)
      scheduler.step()
  init_end_event.record()

  if rank == 0:
    print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
    print(f"{model}")

  if args.save_model:
     dist.barrier()
     states = model.state_dict()
     if rank == 0:
         torch.save(states, "mnist_cnn.pt")
  cleanup()

def mmidas_main(rank, world_size, args):
  ...

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
    parser.add_argument('--mnist', action='store_true', default=True, help='run fsdp on simple mnist example') # TODO
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
    args = parser.parse_args()

    assert not (args.no_parallel and args.multinode), "cannot disable parallelism and enable multinode training"

    torch.manual_seed(args.seed)

    if args.nccl_debug:
      os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker'
    # os.environ['NCCL_IB_DISABLE'] = '0'
    # os.environ['NCCL_IB_HCA'] = 'mlx5_1'
    
    dprint(f"backend: {args.backend}")
    dprint(f"parallel training: {not args.no_parallel}")
    dprint(f"multinode training: {args.multinode}")
    dprint(f"mnist mode: {args.mnist}")
    dprint(f"cpu available: {torch.cpu.is_available()}")
    dprint(f"cuda available: {torch.cuda.is_available()}")
    dprint(f"mps available: {torch.backends.mps.is_available()}")
    if args.no_parallel:
      main(args)
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
      print_gpus()
      dprint(f"world size: {world_size}")
      mp.spawn(fsdp_main,
          args=(world_size, args),
          nprocs=world_size,
          join=True)
      

  

# [] get slurm world size
# [] test cuda device count on sbatch job
