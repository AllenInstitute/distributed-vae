import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms # for datasets
from tqdm import tqdm # for progress bar

import socket
import os
from functools import partial
import time

import torch.distributed as dist # for distributed communication
from torch.utils.data.distributed import DistributedSampler # for distributed data across GPUs
import torch.multiprocessing as mp # for spawning processes on each GPU
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, # FSDP constructor for sharding model parameters
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy # for FSDP configuration

assert torch.cuda.is_available(), "CUDA is not available. You must have CUDA enabled to use distributed training."


class ShallowNet(nn.Module):
  def __init__(self):
    super(ShallowNet, self).__init__()
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
  
# helper function
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# utility functions to initialize distributed training

def get_free_addr():
  return socket.gethostbyname_ex(socket.gethostname())[2][0]
    
def get_free_port(addr):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((addr, 0))
    s.listen(1)
    port = s.getsockname()[1]
  return port

# initialize PyTorch's distributed backend
def setup_distributed(rank, world_size, addr, port):
  os.environ['MASTER_ADDR'] = str(addr) # address that ranks will use to communicate
  os.environ['MASTER_PORT'] = str(port) # port that ranks will use to communicate
  if 'a100' in torch.cuda.get_device_name().lower(): # needed for training with A100's on some SLURM clusters
    os.environ['NCCL_P2P_LVL'] = 'NVL'
  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

# cleanup PyTorch's distributed backend
def cleanup_distributed():
  dist.destroy_process_group()

# we only want to print on the master rank (rank 0) to avoid duplicate logging. This is a convention commonly followed
# when doing distributed training to reduce logging clutter
def is_master():
  return dist.get_rank() == 0

def train_dist(model, train_loader, opt):
  rank = dist.get_rank() # new
  
  total_loss = torch.zeros(1).to(rank) # updated
  num_batches = torch.zeros(1).to(rank) # updated

  model.train()
  bar = tqdm(train_loader, total=len(train_loader)) if is_master() else train_loader # updated: we only want a progress bar on the master rank
  for (data, target) in bar: # updated
    data, target = data.to(rank), target.to(rank) # updated
    opt.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target, reduction='sum')
    memory = torch.cuda.memory_allocated(rank) / 1e6 if torch.cuda.is_available() else 'N/A'
    loss.backward()
    opt.step()
    total_loss += loss
    num_batches += 1
  
  dist.all_reduce(total_loss, op=dist.ReduceOp.SUM) # new
  dist.all_reduce(num_batches, op=dist.ReduceOp.SUM) # new
  avg_loss = total_loss.item() / (num_batches.item() * train_loader.batch_size) # updated
  return {
    'avg_loss': avg_loss,
    'memory': memory
  }

def test_dist(model, test_loader):
  rank = dist.get_rank() # new

  total_loss = torch.zeros(1).to(rank) # updated
  num_batches = torch.zeros(1).to(rank) # updated
  total_correct = torch.zeros(1).to(rank) # updated
  num_datapoints = torch.zeros(1).to(rank) # updated

  model.eval()
  with torch.no_grad():
    bar = tqdm(test_loader, total=len(test_loader)) if is_master() else test_loader
    for (data, target) in bar: # updated: we only want a progress bar on the master rank
      data, target = data.to(rank), target.to(rank)
      output = model(data)
      total_loss += F.nll_loss(output, target, reduction='sum')
      pred = output.argmax(dim=1, keepdim=True)
      total_correct += pred.eq(target.view_as(pred)).sum()
      num_datapoints += len(data)
      num_batches += 1

  dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
  dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
  dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
  dist.all_reduce(num_datapoints, op=dist.ReduceOp.SUM)
  avg_loss = total_loss.item() / (num_batches.item() * test_loader.batch_size)
  accuracy = total_correct.item() / num_datapoints.item()
  return {
    'avg_loss': avg_loss,
    'total_correct': total_correct.item(),
    'num_datapoints': num_datapoints.item(),
    'accuracy': accuracy
  }


def main(rank, world_size, addr, port):
  # training config, same as before
  train_batch_size = 256
  test_batch_size = 1000
  lr = 0.001
  epochs = 10
  
  # new: some config for FSDP
  min_num_params = 20000
  auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
  
  
  setup_distributed(rank, world_size, addr, port) # new: initialize PyTorch's distributed backend

  torch.cuda.set_device(rank) # new: set the device to the current rank

  _transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])
  train_data = datasets.MNIST('../data', train=True, download=True, transform=_transform)
  test_data = datasets.MNIST('../data', train=False, transform=_transform)

  train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank) # new: use DistributedSampler to distribute data
                                                                                     #      across all ranks
  test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank)

  loader_kwargs = {
      'num_workers': 2,
      'pin_memory': True,
      'shuffle': False,
      'drop_last': True,
      'persistent_workers': True # warning: on Allen HPC, disabling this massively slows down training
  }

  train_loader_dist = DataLoader(train_data, batch_size=train_batch_size, sampler=train_sampler, **loader_kwargs) # updated: make sure to pass in the sampler
  test_loader_dist = DataLoader(test_data, batch_size=test_batch_size, sampler=test_sampler, **loader_kwargs)

  # First, we train the shallow model

  shallow_model = ShallowNet().to(rank)
  shallow_model = FSDP(shallow_model, auto_wrap_policy=auto_wrap_policy) # new: wrap the model with FSDP. This will shard the model's parameters across ranks during training
  shallow_optimizer = optim.Adam(shallow_model.parameters(), lr=lr) # make sure to construct the optimizer AFTER wrapping the model with FSDP, as we want to update the 
                                                                    # parameters of the sharded model, not  original model
  if is_master():
    print("> training shallow model\n") # updated: only print on the master rank
  for epoch in range(epochs):
    start_time = time.time()
    train_sampler.set_epoch(epoch) # new: we must ensure each rank works on a different partition of the same batch of data
    train_losses = train_dist(shallow_model, train_loader_dist, shallow_optimizer)

    test_sampler.set_epoch(epoch) # new: we must ensure each rank works on a different partition of the same batch of data
    test_losses = test_dist(shallow_model, test_loader_dist)
    total_time = time.time() - start_time
    if is_master(): # updated: only print on the master rank
      print(f"epoch: {epoch} | train loss: {train_losses['avg_loss']:.4f} | test loss: {test_losses['avg_loss']:.4f} | test acc: {test_losses['total_correct']}/{test_losses['num_datapoints']} ({test_losses['accuracy']:.4f}) | memory: {train_losses['memory']:.2f} MB | time: {total_time:.2f}s")
      print('-' * 100)
    
  # Next, we train the deep model. Most of what applied to the shallow model applies here, too.

  deep_model = DeepNet().to(rank)
  deep_model = FSDP(deep_model, auto_wrap_policy=auto_wrap_policy)
  deep_optimizer = optim.Adam(deep_model.parameters(), lr=lr)
  if is_master():
    print("> training deep model\n")
  for epoch in range(epochs):
    start_time = time.time()
    train_sampler.set_epoch(epoch)
    train_losses = train_dist(deep_model, train_loader_dist, deep_optimizer)

    test_sampler.set_epoch(epoch)
    test_losses = test_dist(deep_model, test_loader_dist)
    total_time = time.time() - start_time
    if is_master():
      print(f"epoch: {epoch} | train loss: {train_losses['avg_loss']:.4f} | test loss: {test_losses['avg_loss']:.4f} | test acc: {test_losses['total_correct']}/{test_losses['num_datapoints']} ({test_losses['accuracy']:.4f}) | memory: {train_losses['memory']:.2f} MB | time: {total_time:.2f}s")
      print('-' * 100)

  cleanup_distributed() # new: cleanup PyTorch's distributed backend to release resources