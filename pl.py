# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import datetime
import functools
import time
from time import sleep
import os

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from pyrsistent import m, v, pmap, pvector, PMap, PVector
PMap.__call__ = lambda self, x: self[x]
PVector.__call__ = lambda self, x: self[x]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import wandb

from my_utils import convert, avg


class PLNet(L.LightningModule):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)
    self.epoch_start_time = None
    self.ep = 0

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
  
  def on_train_epoch_start(self):
    self.epoch_start_time = time.time()

  def on_train_epoch_end(self):
    self.log('epoch_time', time.time() - self.epoch_start_time)
  
  def training_step(self, batch, batch_idx):
    data, target = batch
    output = self(data)
    loss = F.nll_loss(output, target)
    self.log('train_loss', loss, rank_zero_only=True)
    if self.ep > 0:
      self.log('cuda_memory', convert(torch.cuda.memory_allocated(), 'B', 'MB'), sync_dist=True)
    self.ep += 1
    return loss
  
  def validation_step(self, batch, batch_idx):
    data, target = batch
    output = self(data)
    loss = F.nll_loss(output, target)
    self.log('val_loss', loss, rank_zero_only=True)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
  
class PLDeepNet(L.LightningModule):
  def __init__(self):
    super().__init__()
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
    self.epoch_start_time = None
    self.ep = 0

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
  
  def on_train_epoch_start(self):
    self.epoch_start_time = time.time()

  def on_train_epoch_end(self):
    self.log('epoch_time', time.time() - self.epoch_start_time)
  
  def training_step(self, batch, batch_idx):
    data, target = batch
    output = self(data)
    loss = F.nll_loss(output, target)
    self.log('train_loss', loss, rank_zero_only=True)
    if self.ep > 0:
      self.log('cuda_memory', convert(torch.cuda.memory_allocated(), 'B', 'MB'), sync_dist=True)
    self.ep += 1
    return loss
  
  def validation_step(self, batch, batch_idx):
    data, target = batch
    output = self(data)
    loss = F.nll_loss(output, target)
    self.log('val_loss', loss, rank_zero_only=True)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
  
def make_model(s):
  match s:
    case 'net':
      return PLNet()
    case 'deep':
      return PLDeepNet()
    case _:
      raise ValueError(f'unknown model: {s}')
  
def pl_main(args):
  model = make_model(args.model)
  transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
  ])

  dataset1 = datasets.MNIST('../data', train=True, download=True,
                      transform=transform)
  dataset2 = datasets.MNIST('../data', train=False,
                      transform=transform)

  train_kwargs = {'batch_size': args.batch_size,}
  test_kwargs = {'batch_size': args.test_batch_size,}
  cuda_kwargs = {'num_workers': 2,
                  'pin_memory': True,
                  'shuffle': False,
                  'drop_last': True}
  train_kwargs.update(cuda_kwargs)
  test_kwargs.update(cuda_kwargs)
  train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
  
  # remember: you can pass in distributed sampler here, too
  wandb_logger = WandbLogger(project='pl-mnist')
  if isinstance(wandb_logger.experiment.config, dict):
    wandb_logger.experiment.config.update(args)
  trainer = L.Trainer(max_epochs=args.epochs, devices=args.devices, strategy='fsdp', use_distributed_sampler=(not args.no_sampler), logger=wandb_logger)
  trainer.fit(model, train_loader, test_loader)
  
  entity = wandb_logger.experiment.entity
  project = wandb_logger.experiment.project
  run_id = wandb_logger.experiment.id

  if isinstance(entity, str):
    run_path = f"{entity}/{project}/{run_id}"
    print(f'Run path: {run_path}')
    api = wandb.Api()
    
    run = api.run(f"{run_path}")
    history = run.history()
    epoch_times = history['epoch_time'].dropna().values
    memory = history['cuda_memory'].dropna().values
    print(f'Average epoch time: {avg(epoch_times)}')
    print(f'Average memory usage: {avg(memory)}')
    # log metrics
    wandb_logger.log_metrics({'avg_epoch_time': avg(epoch_times), 'avg_memory_usage': avg(memory)})


def make_args(**kwargs):
  return pmap({
    'batch_size': kwargs.get('batch_size', 64),
    'test_batch_size': kwargs.get('test_batch_size', 1000),
    'epochs': kwargs.get('epochs', 10),
    'lr': kwargs.get('lr', 1.0),
    'gamma': kwargs.get('gamma', 0.7),
    'no_cuda': kwargs.get('no_cuda', False),
    'seed': kwargs.get('seed', 1),
    'save_model': kwargs.get('save_model', False),
  })


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
    parser.add_argument('--no-sampler', action='store_true', default=False,
                        help='For not using distributed sampler')
    parser.add_argument('--devices', type=str, default='-1',
                        help='Comma-separated list of device ids to use. Use -1 for all.')
    parser.add_argument('--model', type=str, default='net',
                        help='Model to use: net or deep')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    pl_main(args)

# [] cuda device order
# [] creating CudaAccelerator?
# [] maybe has to do with data loading? test is really slow
# [] try find free network port