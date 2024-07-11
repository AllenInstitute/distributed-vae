# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import datetime
import functools
import time
import os

import lightning as L
from pyrsistent import m, v, pmap, pvector, PMap, PVector
PMap.__call__ = lambda self, x: self[x]
PVector.__call__ = lambda self, x: self[x]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
    print(f"epoch time: {time.time() - self.epoch_start_time}")
  
  def training_step(self, batch, batch_idx):
    data, target = batch
    output = self(data)
    loss = F.nll_loss(output, target)
    self.log('train_loss', loss)
    return loss
  
  def validation_step(self, batch, batch_idx):
    data, target = batch
    output = self(data)
    loss = F.nll_loss(output, target)
    self.log('val_loss', loss)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
  
def pl_main(world_size, args):
  model = PLNet()
  transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
  ])

  dataset1 = datasets.MNIST('../data', train=True, download=True,
                      transform=transform)
  dataset2 = datasets.MNIST('../data', train=False,
                      transform=transform)
  
  sampler1 = torch.utils.data.distributed.DistributedSampler(dataset1, num_replicas=world_size, rank=0)
  sampler2 = ...

  train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
  test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
  cuda_kwargs = {'num_workers': 2,
                  'pin_memory': True,
                  'shuffle': False}
  train_kwargs.update(cuda_kwargs)
  test_kwargs.update(cuda_kwargs)
  train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
  test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
  
  # remember: you can pass in distributed sampler here, too
  trainer = L.Trainer(max_epochs=args.epochs, devices=[0, 1], strategy='fsdp')
  trainer.fit(model, train_loader, test_loader)


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
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # WORLD_SIZE = torch.cuda.device_count()
    WORLD_SIZE = torch.cuda.device_count()
    pl_main(WORLD_SIZE, args)
    # mp.spawn(fsdp_main,
    #     args=(WORLD_SIZE, args),
    #     nprocs=WORLD_SIZE,
    #     join=True)
