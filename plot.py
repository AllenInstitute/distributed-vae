import os
from collections import defaultdict

import torch as T
import numpy as np
import matplotlib.pyplot as plt

def get_run(pth):
  for f in os.listdir(pth):
    if f.endswith('.pt'):
      return f
  raise FileNotFoundError('No .pt file found in {}'.format(pth))

def plot(*metrics, dirn, title=None, save=None, drop_first=2, **kw):
    fig, ax = plt.subplots()
    
    for metric in metrics:
        run_data = defaultdict(list)
        
        for r, name in kw.items():
            pth = f'{dirn}/{r}'
            run_file = get_run(pth)
            data = T.load(f'{pth}/{run_file}', map_location='cpu')
            y = 100 * data[metric]
            run_data[name].append(y[drop_first:])
        
        for name, runs in run_data.items():
            runs = np.array(runs)
            mean = np.mean(runs, axis=0)
            std = np.std(runs, axis=0)
            
            x = np.arange(mean.shape[0])
            ax.plot(x, mean, label=f'{name} - {metric}')
            ax.fill_between(x, mean - std, mean + std, alpha=0.3)

    ax.set(xlabel='Epoch', ylabel='Value')
    
    if not title:
        title = f'{", ".join(metrics)} vs Epoch'
    ax.set_title(title)
    
    ax.legend()
    plt.show()
    if save:
        plt.savefig(save)

# def plot(*metrics, dirn, title=None, save=None, drop_first=2, **kw):
#   fig, ax = plt.subplots()
  
#   for metric in metrics:
#       ys = []
#       for r, name in kw.items():
#           pth = f'{dirn}/{r}'
#           run_file = get_run(pth)
#           data = T.load(f'{pth}/{run_file}', map_location='cpu')
#           y = 100 * data[metric]
#           ys.append(y[drop_first:])
#       ys = np.array(ys)
#       x = np.arange(ys.shape[1])
      
#       for y, name in zip(ys, kw.values()):
#           ax.plot(x, y, label=f'{name} - {metric}')

#   ax.set(xlabel='Epoch', ylabel='Value')
  
#   if not title:
#       title = f'{", ".join(metrics)} vs Epoch'
#   ax.set_title(title)
  
#   ax.legend()
#   plt.show()
#   if save:
#       plt.savefig(save)
