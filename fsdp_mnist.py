# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import builtins
import datetime
from functools import partial, reduce
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

from torch_utils import current_gpu, cpu_count, set_gpu_

def amoritize_import_(m):
   ...


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

def print_gpu():
  print(f"cuda device: {torch.cuda.current_device()}")

def count_gpus():
  return torch.cuda.device_count()

def setup_print_(rank):
  builtins.print = partial(print, f"[rank {rank}]")

def set_environ_flags_():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(12355)
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)

def make_timeout(s):
  return datetime.timedelta(seconds=s)

def setup_process_group_(rank, world_size):
   dist.init_process_group('nccl', rank=rank, world_size=world_size, 
                           timeout=make_timeout(120))
  
def cleanup_process_group_():
    dist.destroy_process_group()

def setup_distributed_(rank, world_size):
  set_environ_flags_()
  setup_process_group_(rank, world_size)

def cleanup_distributed_():
  dist.destroy_process_group()

def is_sum(op):
  return op == 'sum'

def is_product(op):
  return op == 'product'

def is_min(op):
  return op == 'min'

def is_max(op):
  return op == 'max'

def make_reduce_op(op):
  if is_sum(op):
    return dist.ReduceOp.SUM
  elif is_product(op):
    return dist.ReduceOp.PRODUCT
  elif is_min(op):
    return dist.ReduceOp.MIN
  elif is_max(op):
    return dist.ReduceOp.MAX
  else:
    raise ValueError(f"Unknown reduce op: {op}")
  
def all_reduce_(tensor, op='sum'):
  dist.all_reduce(tensor, op=make_reduce_op(op))

def print_train_loss(train_loss, epoch, rank):
   if is_master(rank):
      print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, train_loss[0] / train_loss[1]))


def print_test_loss(test_loss, rank):
    if is_master(rank):
        avg_loss = test_loss[0] / test_loss[2]
        accuracy = 100. * test_loss[1] / test_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            avg_loss, int(test_loss[1]), int(test_loss[2]), accuracy))

def print_epoch(epoch, time, rank):
  if is_master(rank):
      print(f"Epoch {epoch} took {time}sec")

def synchronize_gpus_():
  torch.cuda.synchronize()

def cuda_elapsed_time(start_event, end_event):
  return start_event.elapsed_time(end_event) / 1000

def print_cuda_elapsed_time(time):
  print(f"CUDA event elapsed time: {time}sec")

def print_summary(time, model, rank):
  if is_master(rank):
    print_cuda_elapsed_time(time)
    print(f"{model}")

def make_cuda_event():
  return torch.cuda.Event(enable_timing=True)

def fsdp(*args, **kwargs):
  return FSDP(*args, **kwargs)

def ddp(*args, **kwargs):
  return DDP(*args, **kwargs)

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
  
def make_model(s, rank):
  if s == 'shallow':
    model = Net()
  elif s == 'deep':
    model = DeepNet()
  else:
    raise ValueError(f"Unknown model: {s}")
  return model.to(rank)

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

def string_to_dtype(s):
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
  dtype = string_to_dtype(p)
  return MixedPrecision(
    param_dtype=dtype,
    reduce_dtype=dtype,
    buffer_dtype=dtype
  )

def make_wrap_policy(params):
  return partial(size_based_auto_wrap_policy, min_num_params=params)
   
def is_master(rank):
    return rank == 0

def transform_loader(loader, *funs):
    return reduce(lambda acc, f: f(acc), funs, loader)

def make_pbar(loader, rank, *funs):
    total = len(loader)
    loader = transform_loader(loader, *funs)
    if is_master(rank):
        return tqdm(loader, total=total, unit_scale=True)
    else:
        return loader
    
def make_loader_config(**kwargs):
   return pmap(kwargs)

def make_loader(data, **config):
    return torch.utils.data.DataLoader(data, **config)

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
    
def make_train_loss(model, rank):
    if is_mnist_net(model):
        return torch.zeros(2).to(rank)
    else:
       raise ValueError(f"Unknown model: {model}")

def make_test_loss(model, rank):
    if is_mnist_net(model):
        return torch.zeros(3).to(rank)
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

def make_dist_sampler(data, rank, world_size, shuffle=True):
  return DistributedSampler(data, rank=rank, num_replicas=world_size, shuffle=shuffle)

def record_cuda_event_(event):
  event.record()

def barrier_distributed_():
  dist.barrier()

def lower(s):
  return s.lower()

def is_a100(name):
   return 'a100' in lower(name)

def current_gpu_is_a100():
  return is_a100(torch.cuda.get_device_name())

def save_model_(model, rank):
  if is_master(rank):
    torch.save(model.state_dict(), "mnist_cnn.pt")

def make_scheduler(optimizer, step_size, gamma):
  return StepLR(optimizer, step_size=step_size, gamma=gamma)

def step_(o):
  o.step()

def get_mixed_args(args):
  return args.mixed

def use_compile(args):
  return args.compile

def train_(model, rank, train_loader, optimizer, epoch, sampler=None):
    set_mode_(model, 'train')
    ddp_loss = make_train_loss(model, rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in make_pbar(train_loader, rank, enumerate):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        step_(optimizer)
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    all_reduce_(ddp_loss, op='sum')
    print_train_loss(ddp_loss, epoch, rank)

def test(model, rank, test_loader):
    set_mode_(model, 'eval')
    ddp_loss = make_test_loss(model, rank)
    with torch.no_grad():
      for data, target in make_pbar(test_loader, rank):
        data, target = data.to(rank), target.to(rank)
        output = model(data)
        ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
        ddp_loss[2] += len(data)

    all_reduce_(ddp_loss, op='sum')
    print_test_loss(ddp_loss, rank)

def get_sharding_strategy(args):
  return args.sharding

def compile_model(model):
  return torch.compile(model, mode='max-autotune-no-cudagraphs')

def fsdp_main(rank, world_size, args):
    setup_print_(rank)
    print(f"starting...")
    setup_distributed_(rank, world_size)
    if is_master(rank):
        print("warning: changing matmul precision")
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    transform = make_data_transform()
    train_data = make_train_data(transform)
    test_data = make_test_data(transform)

    train_sampler = make_dist_sampler(train_data, rank, world_size, shuffle=True)
    test_sampler = make_dist_sampler(test_data, rank, world_size, shuffle=False)

    cuda_kwargs = {'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': False,
                    'drop_last': True,
                    'persistent_workers': True, # this is important!!!
                    'prefetch_factor': 2,
                    }
    train_loader_config = make_loader_config(batch_size=args.batch_size, 
                                             sampler=train_sampler, **cuda_kwargs)
    test_loader_config = make_loader_config(batch_size=args.test_batch_size,
                                            sampler=test_sampler, **cuda_kwargs)

    train_loader = make_loader(train_data, **train_loader_config)
    test_loader = make_loader(test_data, **test_loader_config)
    set_gpu_(rank)

    start_event = make_cuda_event()
    end_event = make_cuda_event()

    model = make_model(args.model, rank)
    strat = get_sharding_strategy(args)
    model = fsdp(model, auto_wrap_policy=make_wrap_policy(100), 
                 use_orig_params=use_compile(args), 
                 mixed_precision=make_mixed_precision(args.mixed),
                 sharding_strategy=string_to_sharding_strategy(strat))
    if use_compile(args):
       model = compile_model(model)
    optimizer = make_optimizer(model, args.lr)

    scheduler = make_scheduler(optimizer, step_size=1, gamma=args.gamma)
    record_cuda_event_(start_event)
    for epoch in range(args.epochs):
        t0 = time.time()
        train_(model, rank, train_loader, optimizer, epoch, sampler=train_sampler)
        test(model, rank, test_loader)
        step_(scheduler)
        print_epoch(epoch, time.time() - t0, rank)
    record_cuda_event_(end_event)

    synchronize_gpus_()
    print_summary(cuda_elapsed_time(start_event, end_event), 
                  model, rank)

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        barrier_distributed_()
        save_model_(model, rank)

    cleanup_distributed_()

def make_args(parser, **kwargs):
  return parser.parse_args()
  # return pmap({
  #   'batch_size': kwargs.get('batch_size', 64),
  #   'test_batch_size': kwargs.get('test_batch_size', 1000),
  #   'epochs': kwargs.get('epochs', 10),
  #   'lr': kwargs.get('lr', 1.0),
  #   'gamma': kwargs.get('gamma', 0.7),
  #   'no_cuda': kwargs.get('no_cuda', False),
  #   'seed': kwargs.get('seed', 1),
  #   'save_model': kwargs.get('save_model', False),
  # })

def spawn_processes_(fun, world_size, args):
    mp.spawn(fun, args=(world_size, args), nprocs=world_size, join=True)

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
    parser.add_argument('--mixed', type=str, default='fp32', metavar='M',
                        help='Mixed precision: fp16, bf16, fp32')
    parser.add_argument('--sharding', type=str, default='full', metavar='M',
                        help='Sharding strategy: full, grad-op, no, hybrid, hybrid-zero2')
    args = make_args(parser)
    set_seed_(args.seed)

    world_size = count_gpus()
    spawn_processes_(fsdp_main, world_size, args)

