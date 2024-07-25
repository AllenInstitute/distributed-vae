from torch import nn
from pyrsistent import pmap


class VAE(nn.Module):
  ...

class MMIDAS(nn.Module):
  ...

class Augmenter(nn.Module):
  ...

class Discriminator(nn.Module):
  ...


def make_vae_config(**kw):
  return pmap({
    'input_dim': kw.get('input_dim'),
    'fc_dim': kw.get('fc_dim', 100),
    'state_dim': kw.get('state_dim'),
    'lowD_dim': kw.get('lowD_dim', 10),
    'x_drop': kw.get('x_drop', 0.5),
    's_drop': kw.get('s_drop', 0.2),
    'n_arm': kw.get('n_arm', 2),
    'lam': kw.get('lam', 1),
    'lam_pc': kw.get('lam_pc', 1),
    'tau': kw.get('tau', 0.005),
    'beta': kw.get('beta', 1.),
    'hard': kw.get('hard', False),
    'is_variational': kw.get('is_variational', True),
    'eps': kw.get('eps', 1e-8),
    'ref_prior': kw.get('ref_prior', False),
    'loss_mode': kw.get('loss_mode', 'MSE')
  })


def augment(augmenter, data):
  ...

def make_vae():
  ...

def make_augmenter(dataset):
  ...
