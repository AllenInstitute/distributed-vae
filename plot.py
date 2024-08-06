import os
import re
from functools import lru_cache
from collections import defaultdict, OrderedDict

import torch as T
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import pmap
from plot_config import *

def some(f, xs):
    return next((x for x in xs if f(x)), None)

def starsome(f, xs):
    return next((x for x in xs if f(*x)), None)

def parse_batchnorm(text):
    return (lambda x: x.group(1) if x else None)(re.search(r'use_batchnorm=(True|False)', text))

@lru_cache(maxsize=None)
def parse_run(text):
    return (lambda x: x.group(1) if x else None)(re.search(r'\[r0\] saved to toy-runs/(r\d+)', text))

def get_run(pth):
    return some(lambda f: f.endswith('.pt'), lazy_listdir(pth))

def lazy_listdir(pth):
    return (f.name for f in os.scandir(pth))

@lru_cache(maxsize=None)
def file_to_str(pth):
    with open(pth) as f: return f.read()

def logs_of(run, logs='mnist-logs'):
    return filter(lambda f: parse_run(file_to_str(f'{logs}/{f}')) == run, lazy_listdir(logs))

@lru_cache(maxsize=None)
def get_runs(logs='mnist-logs'):
    return pmap(filter(lambda x: x[1] is not None, 
                  map(lambda f: (f, parse_run(file_to_str(f'mnist-logs/{f}'))),
                      filter(lambda f: f.endswith('.out'), lazy_listdir(logs)))))

@lru_cache(maxsize=None)
def inv(dct):
    return {v: k for k, v in dct.items()}

def no_ext(text):
    return text.split('.')[0]

def parse_title(text, delim='_', kw=None):
    _it = inv(kw).items()
    return pmap(map(lambda x: (lambda kv: (kv[1], x[len(kv[0]):]))(some(lambda kv: x.startswith(kv[0]),_it)),
               no_ext(text).split(delim)[1:]))

def is_config(dct):
    ks = ['model', 'batch_size', 'num_workers', 'epochs', 'gpus',
          'use_dist_sampler', 'sharding_strategy', 'mixed_precision',
          'sync_module_states', 'use_compilation', 'use_orig_params',
          'use_batchnorm']
    return all(k in dct for k in ks)

def parse_title_mnist(text):
    delim = '_'
    kw = pmap({
        'model': 'MODEL',
        'batch_size': 'B',
        'num_workers': 'WORK',
        'epochs': 'E',
        'gpus': 'G',
        'use_dist_sampler': 'DSAMP',
        'sharding_strategy': 'SHARD',
        'mixed_precision': 'PREC',
        'sync_module_states': 'SYNC',
        'use_compilation': 'COMP',
        'use_orig_params': 'ORIG',
    })
    return parse_title(text, delim, kw)

def _plot(*metrics, dirn, style='line', title=None, save=None, drop_first=2,
         scale=1, ylabel=None, use_average=False, **kw):
    fig, ax = plt.subplots()
    
    bar_width = 0.35
    unique_names = list(OrderedDict.fromkeys(kw.values()))
    bar_positions = np.arange(len(unique_names))
    
    for i, metric in enumerate(metrics):
        run_data = defaultdict(list)
        
        for r, name in kw.items():
            pth = f'{dirn}/{r}'
            run_file = get_run(pth)
            data = T.load(f'{pth}/{run_file}', map_location='cpu')
            y = scale * data[metric]
            if isinstance(y, (np.ndarray, T.Tensor, list)):
                y = y[drop_first:]
                if use_average:
                   y = sum(y) / len(y)
            run_data[name].append(y)
        
        if style == 'line':
            for name, runs in run_data.items():
                runs = np.array(runs)
                mean = np.mean(runs, axis=0)
                std = np.std(runs, axis=0)
                
                x = np.arange(mean.shape[0])
                ax.plot(x, mean, label=f'{name} - {metric}')
                ax.fill_between(x, mean - std, mean + std, alpha=0.3)
            
            ax.set(xlabel='Epoch', ylabel=ylabel if ylabel else 'Value')
        
        elif style == 'bar':
            means = []
            stds = []
            for name in unique_names:
                runs = run_data[name]
                if runs:
                    means.append(np.mean(runs))
                    stds.append(np.std(runs))
                else:
                    means.append(0)
                    stds.append(0)
            
            positions = bar_positions + i * bar_width
            ax.bar(positions, means, bar_width, label=metric, yerr=stds, capsize=5)
            
            ax.set_ylabel(ylabel if ylabel else metric)
            ax.set_xticks(bar_positions + bar_width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(unique_names)
        
        else:
            raise ValueError(f"Unsupported style: {style}")

    if not title:
        title = f'{", ".join(metrics)} vs {"Epoch" if style == "line" else "Run"}'
    ax.set_title(title)
    
    ax.legend()
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(save)

def is_run(s):
    return s.startswith('r') and s[1:].isdigit()

@lru_cache(maxsize=None)
def make_config(run, run_dir='toy-runs', log_dir='mnist-logs'):
    return parse_title_mnist(get_run(f'{run_dir}/{run}')).set('use_batchnorm', 
                    parse_batchnorm(file_to_str(f'{log_dir}/{inv(get_runs(log_dir))[run]}')))

def find_runs(config, run_dir='toy-runs', log_dir='mnist-logs'):
    return filter(lambda f: make_config(f, run_dir=run_dir, log_dir=log_dir) == config,
                  filter(is_run, lazy_listdir(run_dir)))


def plot(*metrics, dirn, style='line', title=None, save=None, drop_first=2,
         scale=1, ylabel=None, **kw):
    assert all(is_config(v) for v in kw.values())

# list(find_runs(TWO_GPUS_SHALLOW_FULL.set('use_dist_sampler', 'False')))