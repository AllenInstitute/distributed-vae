# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import builtins
import datetime
from functools import partial, reduce
import signal
import socket
import time
import os

from pyrsistent import m, v, pmap, pvector, PMap, PVector
def set_call_(cls, fun):
    cls.__call__ = fun
set_call_(PMap, lambda self, x: self[x])
set_call_(PVector, lambda self, x: self[x])
from tqdm import tqdm
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
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch._dynamo import OptimizedModule
import matplotlib.pyplot as plt

import wandb

from torch_utils import current_gpu, cpu_count, set_gpu_

def prn(*args, **kw):
  print(*args, **kw)

def is_imported(m):
  return m in globals()

def set_seed_(seed):
   torch.manual_seed(seed)

def is_train(mode):
  return mode == 'train'

def is_eval(mode):
  return mode == 'eval'

def set_mode_(model, mode):
  if is_train(mode):
    model.train()
  elif is_eval(mode):
    model.eval()
  else:
    raise ValueError(f"Unknown mode: {mode}")

def prn_gpu():
  print(f"cuda device: {torch.cuda.current_device()}")

def ct_gpu():
  return torch.cuda.device_count()

def set_prn_(r):
  builtins.print = partial(print, f"[r{r}]")

def ct_dir(pth):
  return sum(os.path.isdir(os.path.join(pth, x)) for x in os.listdir(pth))

def get_free_addr():
  return socket.gethostbyname_ex(socket.gethostname())[2][0]
    
def get_free_port(addr):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((addr, 0))
    s.listen(1)
    port = s.getsockname()[1]
  return port

# TODO: determine master_addr and master_port automatically
def su_env_(a, p):
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['MASTER_ADDR'] = str(a)
    os.environ['MASTER_PORT'] = str(p)
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if 'a100' in torch.cuda.get_device_name().lower():
      os.environ['NCCL_P2P_LEVEL'] = 'NVL' # new! use only for a100's
      print("warning: changing matmul precision")
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
      torch.set_float32_matmul_precision('high')
      # torch.backends.cudnn.benchmark = True


def mk_to(s):
  return datetime.timedelta(seconds=s)

def su_pg_(r, ws):
   dist.init_process_group('nccl', rank=r, world_size=ws, 
                           timeout=mk_to(600))
  
def cu_pg_():
    dist.destroy_process_group()

def su_dist_(r, ws, a=None, p=None):
  su_env_(a, p)
  su_pg_(r, ws)

def cu_dist_():
  dist.destroy_process_group()

def is_sum(op):
  return op == 'sum'

def is_prod(op):
  return op == 'product'

def is_min(op):
  return op == 'min'

def is_max(op):
  return op == 'max'

def make_reduce_op(op):
  if is_sum(op):
    return dist.ReduceOp.SUM
  elif is_prod(op):
    return dist.ReduceOp.PRODUCT
  elif is_min(op):
    return dist.ReduceOp.MIN
  elif is_max(op):
    return dist.ReduceOp.MAX
  else:
    raise ValueError(f"Unknown reduce op: {op}")
  
def all_reduce_(tensor, op='sum'):
  dist.all_reduce(tensor, op=make_reduce_op(op))

def print_train_loss(train_loss, epoch, r):
   if is_master(r):
      print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, train_loss[0] / train_loss[1]))


def print_test_loss(test_loss, r):
    if is_master(r):
        avg_loss = test_loss[0] / test_loss[2]
        accuracy = test_loss[1] / test_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            avg_loss, int(test_loss[1]), int(test_loss[2]), accuracy))

def print_epoch(epoch, time, r):
  if is_master(r):
      print(f"Epoch {epoch} took {time}sec")

def sync_():
  torch.cuda.synchronize()

def cuda_time(start, end):
  return start.elapsed_time(end) / 1000

def print_cuda_time(time):
  print(f"CUDA event elapsed time: {time}sec")

def print_summary(time, model, r):
  if is_master(r):
    print_cuda_time(time)
    print(f"{model}")

def make_cuda_event():
  return torch.cuda.Event(enable_timing=True)

def fsdp(*args, **kw):
  return FSDP(*args, **kw)

def ddp(*args, **kw):
  return DDP(*args, **kw)

signal.signal(signal.SIGINT, lambda _, __: cu_dist_())

class Net(nn.Module):
  def __init__(self, use_batchnorm=False):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.bn1 = nn.BatchNorm1d(128)
    self.fc2 = nn.Linear(128, 10)
    if not use_batchnorm:
       print(f'warning: not using batchnorm')
    self.use_batchnorm = use_batchnorm

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    if self.use_batchnorm:
      x = self.bn1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output

class DeepNet(nn.Module):
  def __init__(self, use_batchnorm=False):
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
    if not use_batchnorm:
      print(f'warning: not using batchnorm')
    else:
      self.bn1a = nn.BatchNorm1d(1000)
      self.bn1b = nn.BatchNorm1d(1000)
      self.bn1c = nn.BatchNorm1d(1000)
    self.use_batchnorm = use_batchnorm

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
    if self.use_batchnorm:
      x = self.bn1a(x)
    x = self.fc1b(x)
    if self.use_batchnorm:
      x = self.bn1b(x)
    x = self.fc1c(x)
    if self.use_batchnorm:
      x = self.bn1c(x)
    x = self.fc1d(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output
  
def make_model(s, r, **kw):
  if s == 'shallow':
    model = Net(use_batchnorm=kw.get('use_batchnorm', False))
  elif s == 'deep':
    model = DeepNet(use_batchnorm=kw.get('use_batchnorm', False))
  else:
    raise ValueError(f"Unknown model: {s}")
  return model.to(r)

def make_optimizer(model, lr):
  if is_mnist_net(model):
    return optim.Adadelta(model.parameters(), lr=lr)
  else:
    raise ValueError(f"Unknown model: {model}")
  
def is_fp16(p):
  return p == 'fp16'

def is_bf16(p):
  return p == 'bf16'

def is_fp32(p):
  return p == 'fp32'

def str2dt(s):
  if s == 'fp16':
    return torch.float16
  elif s == 'bf16':
    return torch.bfloat16
  elif s == 'fp32':
    return torch.float32
  elif s == 'none':
    return None
  else:
    raise ValueError(f"Unknown dtype: {s}")
  
def string_to_sharding_strategy(s):
   if s == 'full':
     print(f'using sharding strategy: full')
     return ShardingStrategy.FULL_SHARD
   elif s == 'grad-op':
      return ShardingStrategy.SHARD_GRAD_OP
   elif s == 'no':
      return ShardingStrategy.NO_SHARD
   elif s == 'hybrid':
      return ShardingStrategy.HYBRID_SHARD
   elif s == 'hybrid-zero2':
      return ShardingStrategy._HYBRID_SHARD_ZERO2
   else:
      raise ValueError(f"Unknown sharding strategy: {s}")
   

def make_mixed_precision(p):
  dtype = str2dt(p)
  if dtype is None:
    return None
  else:
    return MixedPrecision(
      param_dtype=dtype,
      reduce_dtype=dtype,
      buffer_dtype=dtype
    )

def make_wrap_policy(params):
  return partial(size_based_auto_wrap_policy, min_num_params=params)
   
def is_master(r):
    return r == 0

def transform_loader(loader, *funs):
    return reduce(lambda acc, f: f(acc), funs, loader)

def make_pbar(loader, r, *funs):
    total = len(loader)
    loader = transform_loader(loader, *funs)
    if is_master(r):
        return tqdm(loader, total=total, unit_scale=True)
    else:
        return loader
    
def make_loader_config(**kw):
   return pmap(kw)

def make_loader(data, **config):
    return torch.utils.data.DataLoader(data, **config)


# train_loader = DataLoader(data,
#                           use_persistent_workers=True,
#                           sampler=DistributedSampler(...),
#                           ...)

def is_shallow_net(model):
  return isinstance(module(model), Net)

def module(model):
  if is_parallelized(model):
    return module(model.module)
  elif is_compiled(model):
     return module(model._orig_mod)
  else:
    return model

def is_deep_net(model):
  return isinstance(module(model), DeepNet)

def is_fsdp(model):
    return isinstance(model, FSDP)

def is_ddp(model):
    return isinstance(model, DDP)

def is_parallelized(model):
    return is_fsdp(model) or is_ddp(model)

def is_compiled(model):
   return isinstance(model, OptimizedModule)

def is_mnist_net(model):
  return is_shallow_net(module(model)) or is_deep_net(module(model))
    
def make_train_loss(model, r):
    if is_mnist_net(model):
        return torch.zeros(2).to(r)
    else:
       raise ValueError(f"Unknown model: {model}")

def make_test_loss(model, r):
    if is_mnist_net(model):
        return torch.zeros(3, device=r)
    else:
       raise ValueError(f"Unknown model: {model}")
    
def make_data_transform():
  return transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])

def make_train_data(transform):
  return datasets.MNIST('../data', train=True, download=True,
                  transform=transform)

def make_test_data(transform):
  return datasets.MNIST('../data', train=False,
                  transform=transform)

def make_dist_sampler(data, r, ws, shuffle=False):
  return DistributedSampler(data, rank=r, num_replicas=ws, shuffle=shuffle)

def record_cuda_event_(event):
  event.record()

def barrier_():
  dist.barrier()

def lower(s):
  return s.lower()

def is_a100(name):
   return 'a100' in lower(name)

def using_a100():
  return is_a100(torch.cuda.get_device_name())

def save_model_(model, r):
  if is_master(r):
    torch.save(model.state_dict(), "mnist_cnn.pt")

def make_scheduler(optimizer, step_size, gamma):
  return StepLR(optimizer, step_size=step_size, gamma=gamma)

def step_(o):
  o.step()

def get_mixed_args(args):
  return args.mixed

def use_compile(args):
  return args.compile

def convert(x, from_dtype, to_dtype):
  match from_dtype, to_dtype:
    case 'B', 'MB':
      return x / 1024 / 1024
    case _:
      raise ValueError(f"Unknown dtype: {from_dtype}")

def train_(model, r, train_loader, optimizer, epoch, sampler=None, losses=None, epoch_times=None, ws=None, mem=None):
    set_mode_(model, 'train')
    ddp_loss = make_train_loss(model, r)
    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, (data, target) in make_pbar(train_loader, r, enumerate):
        data, target = data.to(r), target.to(r)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        mem.append(convert(torch.cuda.memory_allocated(), 'B', 'MB'))
        loss.backward()
        step_(optimizer)
        ddp_loss[0] += loss.item()
        ddp_loss[1] = train_loader.batch_size

    ddp_loss[0] = ddp_loss[0] / (batch_idx + 1)
    all_reduce_(ddp_loss, op='sum')
    losses[epoch] = (ddp_loss[0] / ddp_loss[1])
    print_train_loss(ddp_loss, epoch, r)


def train_no_cpu_(model, r, train_loader, optimizer, epoch, sampler=None, losses=None, epoch_times=None):
   pass

def test(model, r, test_loader, val_losses, epoch):
    set_mode_(model, 'eval')
    ddp_loss = make_test_loss(model, r)
    with torch.no_grad():
      for batch_indx, (data, target) in make_pbar(test_loader, r, enumerate):
        data, target = data.to(r), target.to(r)
        output = model(data)
        ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True) 
        ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
        ddp_loss[2] = test_loader.batch_size

    ddp_loss[0] = ddp_loss[0] / (batch_indx + 1)
    all_reduce_(ddp_loss, op='sum')
    val_losses[epoch] = (ddp_loss[0] / ddp_loss[2])
    print_test_loss(ddp_loss, r)

def get_sharding_strategy(args):
  return args.sharding

def compile_model(model):
  return torch.compile(model)

def ct_gpu():
  return torch.cuda.device_count()

def use_orig_params(args):
   return args.use_orig_params

def count_cpus_():
  if hasattr(os, 'sched_getaffinity'):
    return len(os.sched_getaffinity(0))
  else:
    return os.cpu_count()

def count_num_workers(args):
  if args.num_workers == -1:
    return count_cpus_() // 2
  else:
    return args.num_workers
  
def get_prefetch_factor(args):
  if args.prefetch_factor == -1:
    if count_num_workers(args) == 0:
      return None
    else:
      return 2
  else:
    return args.prefetch_factor

def main(r, ws, args):
    set_prn_(r)
    prn(f"starting...")
    su_dist_(r, ws, args.addr, args.port)
    # if is_master(r):
    #     print("warning: changing matmul precision")


    transform = make_data_transform()
    train_data = make_train_data(transform)
    test_data = make_test_data(transform)

    if args.use_dist_sampler:
      train_sampler = make_dist_sampler(train_data, r, ws, shuffle=False)
      test_sampler = make_dist_sampler(test_data, r, ws, shuffle=False)
    else:
      if is_master(r):
        print('warning: not using distributed sampler')
      train_sampler = None
      test_sampler = None

    cuda_kw = {'num_workers': count_num_workers(args),
                    'pin_memory': True,
                    'shuffle': False,
                    'drop_last': True,
                    'persistent_workers': count_num_workers(args) > 0,
                    'prefetch_factor': get_prefetch_factor(args),
                    }
    train_loader_config = make_loader_config(batch_size=args.batch_size, 
                                             sampler=train_sampler, **cuda_kw)
    test_loader_config = make_loader_config(batch_size=args.test_batch_size,
                                            sampler=test_sampler, **cuda_kw)

    train_loader = make_loader(train_data, **train_loader_config)
    test_loader = make_loader(test_data, **test_loader_config)
    set_gpu_(r)

    start_event = make_cuda_event()
    end_event = make_cuda_event()

    model = make_model(args.model, r, use_batchnorm=args.use_batchnorm)
    strat = get_sharding_strategy(args)
    if ws > 1:
      if strat == 'ddp':
        if is_master(r): 
          print("using ddp")
        model = DDP(model, device_ids=[r], output_device=r)
      else:
        if is_master(r):
          print("using fsdp")
        model = fsdp(model, auto_wrap_policy=make_wrap_policy(20000), 
                    sharding_strategy=string_to_sharding_strategy(strat),
                     use_orig_params=args.use_orig_params or args.compile,
                     sync_module_states=args.sync,
                     mixed_precision=make_mixed_precision(args.mixed)
                    )
      # model = DDP(model, device_ids=[r], output_device=r)
    if use_compile(args):
       model = compile_model(model)
    optimizer = make_optimizer(model, args.lr)

    losses = torch.empty(args.epochs, device=r)
    epoch_times = torch.empty(args.epochs, device=r)
    mem = []

    val_losses = torch.empty(args.epochs, device=r)
    scheduler = make_scheduler(optimizer, step_size=1, gamma=args.gamma)
    record_cuda_event_(start_event)
    for epoch in range(args.epochs):
        t0 = time.time()
        train_(model, r, train_loader, optimizer, epoch, sampler=train_sampler, losses=losses, epoch_times=epoch_times, ws=ws, mem=mem)
        test(model, r, test_loader, val_losses, epoch)
        step_(scheduler)
        _t = time.time() - t0
        print_epoch(epoch, _t, r)
        epoch_times[epoch] = _t
    record_cuda_event_(end_event)

    barrier_()
    if is_master(r):
    # if False:
      print(f"losses: {losses}")
      print(f"val_losses: {val_losses}")
      print(f"mem: {max(mem)}")
      print(f"epoch times: {epoch_times[1:]}")

      _fldr = 'toy-runs'
      d = f"{_fldr}/r{ct_dir(_fldr)}"
      os.makedirs(d, exist_ok=True)

      plt.plot(losses[1:].cpu().numpy(), label='train')
      plt.plot(val_losses[1:].cpu().numpy(), label='test')
      plt.legend()
      plt.savefig(f'{d}/{mk_filen("LossesPlot")}.png')
      plt.close()

      plt.plot(epoch_times[1:].cpu().numpy(), label='epoch_times')
      plt.legend()
      plt.savefig(f'{d}/{mk_filen("EpochTimesPlot")}.png')
      plt.close()

      torch.save({
        'losses': losses,
        'val_losses': val_losses,
        'epoch_times': epoch_times,
        'ws': ws,
        'args': args,
        'mem': mem,
        'mem_summary': max(mem),
        'str(model)': str(model),
        'sharding_strat': strat,
        'prec': args.mixed,
        'orig': args.use_orig_params,
        'sync': args.sync,
        'compile': args.compile
        # 'gpu': torch.cuda.get_device_name(),
      }, f'{d}/{mk_filen("Run", model=args.model, b=args.batch_size, work=args.num_workers, E=args.epochs, g=args.gpus, dsamp=args.use_dist_sampler, shard=strat, prec=args.mixed, sync=args.sync, comp=args.compile, orig=args.use_orig_params)}.pt')
      print(f"saved to {d}")

    sync_()
    print_summary(cuda_time(start_event, end_event), 
                  model, r)

    if args.save_model:
        # use a barrier to make sure training is done on all rs
        barrier_()
        save_model_(model, r)

    cu_dist_()

def make_args(parser, **kw):
  return parser.parse_args()
  # return pmap({
  #   'batch_size': kw.get('batch_size', 64),
  #   'test_batch_size': kw.get('test_batch_size', 1000),
  #   'epochs': kw.get('epochs', 10),
  #   'lr': kw.get('lr', 1.0),
  #   'gamma': kw.get('gamma', 0.7),
  #   'no_cuda': kw.get('no_cuda', False),
  #   'seed': kw.get('seed', 1),
  #   'save_model': kw.get('save_model', False),
  # })

def spawn_(fun, ws, args):
    mp.spawn(fun, args=(ws, args), nprocs=ws, join=True)

def ct_gpu_args(args):
  if args.gpus == -1:
      return ct_gpu()
  else:
      return args.gpus

def no(s, *c):
  return reduce(lambda acc, x: acc.replace(x, ''), c, s)

def up(s):
  return s.upper()
  
def mk_filen(nm, **kw):
  return reduce(lambda acc, x: f"{acc}_{up(no(str(x), '-', '.'))}{kw[x]}", kw, nm)

# TODO
def sv_cfg(name, **kw):
  with open(f'configs/{mk_filen(name, **kw)}.txt', 'w') as f:
    for k, v in kw.items():
      f.write(f"{k}: {v}\n")

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
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
    parser.add_argument('--model', type=str, default='shallow', metavar='M',
                        help='Model to use: shallow, deep')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Compile the model')
    parser.add_argument('--mixed', type=str, default='none', metavar='M',
                        help='Mixed precision: fp16, bf16, fp32')
    parser.add_argument('--sharding', type=str, default='full', metavar='M',
                        help='Sharding strategy: full, grad-op, no, hybrid, hybrid-zero2')
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--use_orig_params',  default=False, action='store_true')
    parser.add_argument('--sync', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--prefetch_factor', type=int, default=-1)
    parser.add_argument('--use_batchnorm', default=False, action='store_true')
    parser.add_argument('--use_dist_sampler', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    args = make_args(parser)
    set_seed_(args.seed)

    ws = ct_gpu_args(args)
    prn(f'ws: {ws}')
    args.addr = get_free_addr()
    args.port = get_free_port(args.addr)
    args.num_workers = count_num_workers(args)
    args.gpus = ws
    args.prefetch_factor = get_prefetch_factor(args)
    prn(args)
    spawn_(main, ws, args)



# model = nn.SyncBathnorm.sync_batchnorm(model) 
# (64, 1, 28, 28)
# (64)


# no dataloader:
  # fsdp(20000):
  # ddp
  # no:

# dataloader:
  # fsdp(200000): ~3.75s
  # fsdp(20000): ~3.9s
    # fsdp(20000, limit_all_fathers=False): ~3.75s
    # fsdp(20000, forward_prefetch=True): ~3.85s
    # fsdp(20000, limit_all_gathers=False, forward_prefetch=True): ~4s
  # fsdp(2000): ~4.3s
  # fsdp(100): ~5.9s
  # ddp: ~2.75s
  # no: ~3s

  # deep
  # fsdp(200000): ~88s
  # fsdp(20000): ~90s


# python fsdp_mnist.py --epochs 1000 --model shallow --gpus 2 --batch-size 256
  # [x] with distributeds sampler 
  # [] without distributed sampler
  # [] without sharding
  # [] aggregate plots
  # [] deep
