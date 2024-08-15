import os
import re
from functools import lru_cache, reduce as rdc
from itertools import accumulate as acm
from collections import defaultdict, OrderedDict

import torch as T
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import pmap, v
from plot_config import *

def some(f, xs):
    return next((x for x in xs if f(x)), None)

def starsome(f, xs):
    return next((x for x in xs if f(*x)), None)

def reUnwrap(r):
    return r.group(1) if r else None

def reSearch(r, s):
    return reUnwrap(re.search(r, s))

def parseBatchnorm(s):
    return reSearch(r'use_batchnorm=(True|False)', s)

@lru_cache(maxsize=None)
def parseRun(s):
    return reSearch(r'\[r0\] saved to toy-runs/(r\d+)', s)

def runFile(fldr):
    return some(lambda f: f.endswith('.pt'), seqDir(fldr))

def seqDir(fldr):
    return (f.name for f in os.scandir(fldr))

def getRun(pth):
    return parseRun(fileToStr(pth))

@lru_cache(maxsize=None)
def fileToStr(pth):
    with open(pth) as f: return f.read()

def logs_of(run, logs='mnist-logs'):
    return filter(lambda f: getRun(f'{logs}/{f}') == run, seqDir(logs))

def outFiles(logs='mnist-logs'):
    return filter(lambda f: f.endswith('.out'), seqDir(logs))

def noNone(assocs):
    return filter(lambda x: x[1] is not None, assocs)

def mapCons(f, xs):
    return map(lambda x: (x, f(x)), xs)

@lru_cache(maxsize=None)
def runFiles(logs='mnist-logs'):
    return pmap(noNone(mapCons(lambda f: getRun(f'{logs}/{f}'), outFiles(logs))))

@lru_cache(maxsize=None)
def dctInv(dct):
    return {v: k for k, v in dct.items()}

def noExt(text):
    return text.split('.')[0]

def parseTitle(t, env=None):
    def _parse(t, sep):
        return noExt(t).split(sep)[1:]

    def _get_evaluator(x, env):
        return some(lambda kv: x.startswith(kv[0]), env.items())

    def _eval(x, env):
        return (lambda kv: (kv[1], x[len(kv[0]):]))(_get_evaluator(x, env))

    return pmap(map(partial(_eval, env=dctInv(env)), _parse(t, env.sep)))

def parseTitleMNIST(t):
    env = pmap({
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
        'sep': '_'
    })
    return parseTitle(t, env)

def filterEq(a, xs):
    return filter(lambda x: x == a, xs)

def filterEqSnd(a, xs):
    return filter(lambda x: x[1] == a, xs)

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
            run_file = runFile(pth)
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

def isRun(s):
    return s.startswith('r') and s[1:].isdigit()

@lru_cache(maxsize=None)
def mkCfg(run, run_dir='toy-runs', log_dir='mnist-logs'):
    return parseTitleMNIST(runFile(f'{run_dir}/{run}')).set('use_batchnorm', 
                    parseBatchnorm(fileToStr(f'{log_dir}/{dctInv(runFiles(log_dir))[run]}')))

def fsts(xs):
    return map(lambda x: x[0], xs)

# warning: bug here in call to mkCfg
def findRuns(config, run_dir='toy-runs', log_dir='mnist-logs'):
    return fsts(filterEqSnd(config, mapCons(mkCfg, filter(isRun, seqDir(run_dir)))))

def thk0(x):
    return lambda _: x

def mapV(f, assocs):
    return map(lambda x: (x[0], f(x[1])), assocs)

def conj(x, xs):
    return xs.append(x)

def swap(fn):
    return lambda x, y: fn(y, x)

def plot(*metrics, dirn='toy-runs', style='line', title=None, save=None, drop_first=2,
         scale=1, ylabel=None, **kw):
    _kw = acm(map(lambda x: mapCons(thk0(x[0]), x[1]), mapV(findRuns, kw.items())),
              lambda acc, x: rdc(swap(conj), x, acc),
              initial=v())
    _plot(*metrics, dirn=dirn, style=style, title=title, save=save,
          drop_first=drop_first, scale=scale, ylabel=ylabel, 
          **dict(list(_kw)[-1]))