import pprint
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
  always_wrap_policy,
  size_based_auto_wrap_policy,
  enable_wrap,
  wrap,
)
from datetime import timedelta
from tqdm import trange
import wandb
import mmidas
import string
import random
import threading
import time
import matplotlib.pyplot as plt

class MemoryLogger:
  def __init__(self, world_size, interval=0.005, run=None):
    self.memory_allocated = [[] for _ in range(world_size)]
    self.running = True
    self.interval = interval
    self.world_size = world_size
    self.run = run

  def log_memory(self):
    start = time.time()
    while self.running:
      for r in range(self.world_size):
        self.memory_allocated[r].append(MB(torch.cuda.memory_allocated(r)))
        if self.run is not None:
          self.run.log({f'gpu {r} memory': self.memory_allocated[r][-1]})
      time.sleep(self.interval)
    
  def start(self):
    self.thread = threading.Thread(target=self.log_memory)
    self.thread.start()

  def stop(self):
    self.running = False
    self.thread.join()

def get_plotter(s):
  match s:
    case 'line':
      return plt.plot
    case 'scatter':
      return plt.scatter
    case _:
      raise ValueError(f"invalid plot type: {s}")

def truncate(lst, n):
  ret = []
  for i in range(min(n, len(lst))):
    ret.append(lst[(len(lst) // n) * i])
  return ret
  

def MB(bytes):
  return bytes / 1024**2

def random_string(n):
  return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

# %%
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

def plot(*data, plot_t, xlabel, ylabel, title, legend, fname, folder=''):
  plotter = get_plotter(plot_t)
  for d in data:
    plotter(range(len(d)), d)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  if len(data) > 1:
    plt.legend([f'{legend} {i}' for i in range(len(data))])
  if not os.path.exists(folder) and not (folder == ''): 
    os.makedirs(folder)
  plt.savefig(f"{folder}/{fname}")
  dprint(f"> saved plot to {folder}/{fname}")
  plt.close()


def trivial(rank, world_size, args):
  parallel = not args.no_parallel

  setup(rank, world_size, parallel=parallel, is_multinode=args.multinode)
  dprint(f"> starting trivial test on rank {rank}")
  if parallel:
    local_rank = rank if not args.multinode else int(os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(local_rank)
  for _ in range(args.repeat):  
    t = torch.tensor([1, 2, 3], device=rank)
    dprint(f"rank {rank} - before reduce: {t}")
    if parallel:
      dist.all_reduce(t, op=dist.ReduceOp.SUM)
      dprint(f"rank {rank} - after reduce: {t}")
  cleanup(parallel=parallel)

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
  dprint(f"rank {rank} - master addr: {os.environ['MASTER_ADDR']}")
  dprint(f"rank {rank} - master port: {os.environ['MASTER_PORT']}")
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
    
# %%
# def mmidas_data

# %% 
def mmidas_dataloaders(batch_size):
  ...
  
# %%
def make_model(name, parallel, rank=None, **config):
  model = None
  dest = None
  dprint(f"> rank {rank} - making model: {name}")
  match name:
    case 'net':
      model = Net()
    case 'mmidas':
      raise NotImplementedError # TODO
    case _:
      raise ValueError(f"invalid model: {name}")
  return model


# %%
def transform_model(model, is_fsdp, is_jit, rank, wrap, min_params=1000, offload=None):
  if is_fsdp:
    if rank == 0:
      dprint(f"> transforming: fsdp")
    my_auto_wrap_policy = None
    match wrap:
      case 'size_based':
        my_auto_wrap_policy = functools.partial(
          size_based_auto_wrap_policy, min_num_params=min_params
        )
        dprint(f"\trank {rank} - > applying sized based wrap policy (min params: {min_params})")
      case 'always':
        my_auto_wrap_policy = always_wrap_policy
        dprint(f"\trank {rank} - > applying always wrap policy")
      case 'none':
        my_auto_wrap_policy = None
      case _:
        raise ValueError(f"invalid wrap policy: {wrap}")
    cpu_offload = CPUOffload(offload) if offload else None
    if offload: dprint(f"\trank {rank} - > offloading to cpu: {offload}")
    # maybe pass in device_id
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=cpu_offload, use_orig_params=is_jit)
  if is_jit:
    dprint(f"> transforming: jit")
    model = torch.compile(model)
  return model

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None, parallel=True, run=None, print_loss=True, is_reduce=True):
  model.train()
  ddp_loss = torch.zeros(2).to(rank)
  if sampler:
    sampler.set_epoch(epoch)
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(rank), target.to(rank) # TODO
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target, reduction='sum')
    loss.backward()
    optimizer.step()
    ddp_loss[0] += loss.item()
    ddp_loss[1] += len(data)
  if parallel and is_reduce:
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
  if print_loss:
    dprint('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
  if run is not None:
    run.log({'train_loss': ddp_loss[0] / ddp_loss[1],
             'gpu max mem': torch.cuda.max_memory_allocated(rank) / 1024**2,})

def test(model, rank, world_size, test_loader, parallel=True, print_loss=True, is_reduce=True):
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
    if parallel and is_reduce:
      dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if print_loss:
      test_loss = ddp_loss[0] / ddp_loss[2]
      dprint('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
          100. * ddp_loss[1] / ddp_loss[2]))

def fsdp_main(rank, world_size, args):
  parallel = not args.no_parallel
  is_fsdp = not args.no_fsdp
  print_train_loss = 'train' not in args.no_loss
  print_test_loss = 'test' not in args.no_loss

  if parallel:
    assert torch.cuda.is_available()

  _name = f"({torch.cuda.get_device_name(rank)})" if rank == 'cuda' or isinstance(rank, int) else ''
  dprint(f"> training on {rank2dev(rank)} {_name} (host: {os.uname().nodename})")

  setup(rank, world_size, args.backend, args.multinode, args.timeout, parallel=parallel)

  train_loader, test_loader, sampler = make_data_loaders(task=args.task, parallel=(parallel and not args.no_sampler),
                                                batch_size=args.batch_size,
                                                test_batch_size=args.test_batch_size, rank=rank, world_size=world_size)
  if parallel:
    local_rank = rank if not args.multinode else rank - args.gpus_per_node * (rank // args.gpus_per_node)
    torch.cuda.set_device(local_rank)

  # if args.time_cuda:
  init_start_event = torch.cuda.Event(enable_timing=True)
  init_end_event = torch.cuda.Event(enable_timing=True)

  model = make_model(args.model, parallel, rank=rank)
  model = model.to(rank)
  model = transform_model(model, (is_fsdp and parallel), args.jit, rank, args.wrap, args.min_params, args.cpu_offload)
  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

  run = None
  if args.wandb:
    wandb.require('service')
    run = wandb.init(project='dist-mmidas',
                    group=f"{args.task}-{args.id}")
    run.watch(model, log='all', log_freq=100, idx=(rank if type(rank) == int else None), log_graph=True)

  # if args.time_cuda:
  init_start_event.record()
  
  if (not args.no_train) or (not args.no_test):
    if args.record_memory_history:
      torch.cuda.memory._record_memory_history()

    if args.plot_memory:
      mlogger = MemoryLogger(world_size, interval=args.interval, run=run)
      if rank == 0 or not parallel:
        mlogger.start()


    pbar = trange(args.epochs, colour='red') if rank == 0 or not parallel else range(args.epochs)
    for epoch in pbar:
      if not args.no_train:
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler, parallel=parallel, run=run, print_loss=(rank == 0 or not parallel) and print_train_loss)
      if not args.no_test:
        test(model=model, rank=rank, world_size=world_size, test_loader=test_loader, parallel=parallel, print_loss=(rank == 0 or not parallel) and print_test_loss)
        scheduler.step()

    
    if args.plot_memory and (rank == 0 or not parallel):
      memory_allocated = mlogger.memory_allocated
      mlogger.stop()
      for r in range(world_size):
        dprint(f"rank {r} - memory allocated: {truncate(memory_allocated[r], 30)}")
      plot(*memory_allocated, 
           plot_t=args.plot_type, 
           xlabel=f'time ({args.interval}s)', 
           ylabel='memory (MB)', 
           title='CUDA Memory Usage', 
           legend='rank',
           fname=f"{time.time()}-{args.id}-{rank}.png",
           folder='memory-plots')
      # plotter = get_plotter(args.plot_type)
      # for r in memory_allocated:
      #   plotter(range(len(r)), r)
      # plt.xlabel(f'time ({args.interval}s)')
      # plt.ylabel('memory (MB)')
      # plt.title('CUDA Memory Usage')
      # plt.legend([f'rank {r}' for r in range(world_size)])
      # if not os.path.exists("memory-plots"):
      #   os.makedirs("memory-plots")
      # fname = f"memory-plots/{time.time()}-{args.id}-{rank}.png"
      # plt.savefig(fname)
      # plt.close()
      # dprint(f"> saved memory plot to {fname}")

  # if args.time_cuda:

  init_end_event.record()

  if args.record_memory_history:
    if not os.path.exists("memory-snapshots"):
      os.makedirs("memory-snapshots")
    fname = f"memory-snapshots/{time.time()}-{args.id}-{rank}.pickle"
    torch.cuda.memory._dump_snapshot(fname)
    dprint(f"> saved memory snapshot to {fname}")

  # if args.time_cuda:
  if rank == 0 or rank == 'cuda':
    torch.cuda.synchronize()
    print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
  if rank == 0 or not parallel:
    print(f"{model}")

  # save(save_model=args.save_model, parallel=parallel, rank=rank, model=args.)
  if args.save_model:
     if parallel:
      dist.barrier() # TODO: might give bugs
     states = model.state_dict()
     if rank == 0 or not parallel:
         torch.save(states, "mnist_cnn.pt")
  cleanup(parallel=parallel)

def mnist_main(rank, world_size, args):
  dprint(f"> rank {rank} (example) - starting...")

  parallel = not args.no_parallel

  setup(rank, world_size)

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
  cuda_kwargs = {'num_workers': 2,
                  'pin_memory': True,
                  'shuffle': False}
  train_kwargs.update(cuda_kwargs)
  test_kwargs.update(cuda_kwargs)

  train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

  my_auto_wrap_policy = functools.partial(
      size_based_auto_wrap_policy, min_num_params=args.min_params
  )
  torch.cuda.set_device(rank)
  init_start_event = torch.cuda.Event(enable_timing=True)
  init_end_event = torch.cuda.Event(enable_timing=True)
  
  model = Net().to(rank)

  model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, cpu_offload=CPUOffload(args.cpu_offload) if args.cpu_offload else None)

  optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

  # ***** this is mnist_main ******
  if args.wandb:
    wandb.require('service')
    run = None
    run = wandb.init(project='dist-mmidas',
                    group=f"{args.task}-{args.id}")
    run.watch(model, log='all', log_freq=100, idx = (rank if type(rank) == int else None), log_graph=True)

  scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
  init_start_event.record()
  for epoch in trange(1, args.epochs + 1):
      assert parallel
      train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1, parallel=parallel, run=run)
      test(model, rank, world_size, test_loader, parallel=parallel)
      scheduler.step()

  init_end_event.record()

  if rank == 0:
      print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
      print(f"{model}")

  if args.save_model:
      # use a barrier to make sure training is done on all ranks
      dist.barrier()
      states = model.state_dict()
      if rank == 0:
          torch.save(states, "mnist_cnn.pt")

  cleanup()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 128)')
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
    parser.add_argument('--nccl-debug', action='store_true', default=False, help='enable NCCL debugging')
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend (default: nccl)')
    parser.add_argument('--multinode', action='store_true', default=False, help='enable multinode training') # TODO
    parser.add_argument('--no-parallel', action='store_true', default=False, help='disable parallelism')
    parser.add_argument('--min-params', type=int, default=20000, help='minimum number of parameters to wrap')
    parser.add_argument('--no-wrap', action='store_true', default=False, help='disable wrapping')
    parser.add_argument('--cpu-offload', action='store_true', default=False, help='enable CPU offload')
    parser.add_argument('--timeout', type=int, default=120, help='timeout for distributed ops (default: 120)')
    parser.add_argument('--no-train', action='store_true', default=False, help='disable training')
    parser.add_argument('--no-test', action='store_true', default=False, help='disable testing')
    parser.add_argument('--trivial', action='store_true', default=False, help='run trivial test')
    parser.add_argument('--implementation', type=str, default='torch', help='implementation, torch or jax (default: torch)') # TODO
    parser.add_argument('--mixed', action='store_true', default=False, help='use mixed precision for faster training') # TODO
    parser.add_argument('--jit', action='store_true', default=False, help='jit compile the model for faster training') # TODO
    parser.add_argument('--wandb', action='store_true', default=False, help='log to wandb') # TODO
    parser.add_argument('--device', type=str, default='cuda', help='device to use (default: cuda)')
    parser.add_argument('--task', type=str, default='mmidas', help='name of the model to train (default: mmidas-smartdq). Options: mmidas-smartseq, mmidas-10x, mnist, trivial') # TODO
    parser.add_argument('--no_fsdp', action='store_true', default=False, help='disable fsdp')
    parser.add_argument('--time_cuda', action='store_true', default=False, help='time cuda ops') # TODO
    parser.add_argument('--world-size', type=int, default=-1, help='world size for distributed training (default: -1)')
    parser.add_argument('--example', action='store_true', default=False, help='run mnist example code')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat trivial test')
    parser.add_argument('--no-sampler', action='store_true', default=False, help='disable distributed sampler')
    parser.add_argument('--backward_prefetch', action='store_true', default=False, help='enable backward prefetching') # TODO
    parser.add_argument('--wrap', type=str, default='size_based', help='fsdp wrap policy (default: size_based). Options: size_based, always, none') # TODO
    parser.add_argument('--default', action='store_true', default=False, help='use default settings') # TODO
    parser.add_argument('--plot-type', type=str, choices=['line', 'scatter'], default='line', help='plot type (default: line)')
    parser.add_argument('--plot-memory', action='store_true', default=False, help='plot cuda memory') # TODO
    parser.add_argument('--plot-time', action='store_true', default=False, help='plot elapsed time') # TODO
    parser.add_argument('--plot-backend', type=str, default='matplotlib', help='plot backend (default: wandb). Options: matplotlib, plotly, seaborn') # TODO
    parser.add_argument('--model', type=str, default='net', help='model to train (default: net). Options: mmidas, net') # TODO
    parser.add_argument('--parallel', type=str, default='fsdp', help='parallel training method (default: fsdp). Options: fsdp, ddp, none') # TODO
    parser.add_argument('--no-loss', nargs='+', default=[], help='losses to disable (default: [])')
    parser.add_argument('--no-reduce', action='store_true', default=False, help='disable reduce ops') # TODO
    parser.add_argument('--record-memory-history', action='store_true', default=False, help='record memory history')
    parser.add_argument('--tensor-parallel', action='store_true', default=False, help='use tensor parallelism') # TODO
    parser.add_argument('--interval', type=float, default=0.005, help='memory logging interval (default: 0.005)')

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
    args = parser.parse_args()

    if args.parallel == 'none':
      args.no_parallel = True
    args.id = wandb.util.random_string(4) if args.wandb else random_string(4)

    for arg in sorted(vars(args)):
      dprint(f"{arg}: {getattr(args, arg)}")
    dprint()

    # TODO: add other args checking
    assert not (args.no_parallel and args.multinode), "cannot disable parallelism and enable multinode training"
    assert not (not args.no_parallel and (args.device == 'cpu' or args.device == 'mps')), "cannot disable parallelism and use cpu or mps"

    torch.manual_seed(args.seed)

    if args.nccl_debug:
      os.environ['NCCL_DEBUG'] = 'INFO'
    
    dprint('available devices:')
    dprint(f"cpu: {torch.cpu.is_available()}")
    dprint(f"cuda: {torch.cuda.is_available()}")
    dprint(f"mps: {torch.backends.mps.is_available()}")
    dprint()

    dprint(f"visible cuda devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    if args.task == 'trivial':
      if args.no_parallel:
        rank = args.device
        dprint(f"rank: {rank}")
        trivial(rank, 1, args)
      elif args.multinode:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ['SLURM_PROCID'])

        # world_size = int(os.environ['SLURM_NTASKS'])
        
        dprint(f"slurm gpus on node: {os.environ['SLURM_GPUS_ON_NODE']}")
        dprint(f"world size: {world_size}")
        # dprint(f"local world size: {os.environ['LOCAL_WORLD_SIZE']}")
        dprint(f"rank: {rank}")
        dprint(f"local rank: {os.environ['SLURM_LOCALID']}")
        trivial(rank, world_size, args)
      else:
        world_size = torch.cuda.device_count() if args.world_size == -1 else args.world_size
        dprint(f"world size: {world_size}")
        mp.spawn(trivial,
                args=(world_size, args),
                nprocs=world_size,
                join=True)
    elif args.no_parallel:
      world_size = 1 if args.world_size == -1 else args.world_size
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
      world_size = torch.cuda.device_count() if args.world_size == -1 else args.world_size
      dprint(f"world size: {world_size}")
      mp.spawn(mnist_main if args.example else fsdp_main,
          args=(world_size, args),
          nprocs=world_size,
          join=True)

# TODO
# [] --world-size flag
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


# distributed sampler: slowed process, gave zero cuda usage

# mnist plot 
# 1 node, 1-4gpu, w/wo sampler
# [] plot average gpu usage 
# [] plot average elapsed time 


# min-num-params
# [] 1000
# [] 10000
# [] 20000


# if time:
# [] test w/ multiple nodes

# remember: srun vs sbatch vs torchrun
# [] remove seconds at beginning from plot

# [] disable some logging when python -O
# [] fix plot legend