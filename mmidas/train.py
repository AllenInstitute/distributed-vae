import torch

from pyrsistent import pmap

def has_gpu():
  return torch.cuda.is_available()

def has_mps():
  return torch.backends.mps.is_available()

def default_device():
  if has_gpu():
    return 'cuda'
  elif has_mps():
    return 'mps'
  else:
    return 'cpu'

def make_train_config(**kwargs):
  return pmap({
    'use_save': kwargs.get('save', True),
    'folder': kwargs.get('folder', ''),
    'aug_file': kwargs.get('aug_file', ''),
    'device': kwargs.get('device', default_device()),
  })


def make_data():
  ...

def make_loaders():
  ...

def train_vae_():
  if is_parallel(vae):
    return _train_vae_dist_()
  else:
    return _train_vae_()

def _train_vae_():
  ...

def _train_vae_dist_():
  ...

def val_vae():
  ...

def val_vae_dist():
  ...

def prune():
  ...

def train_augmenter():
  ...

def train_joint():
  ...