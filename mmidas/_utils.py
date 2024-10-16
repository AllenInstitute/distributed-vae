from functools import reduce, wraps
import warnings
import time

import torch as th
import numpy as np
from scipy.optimize import linear_sum_assignment

# types: labels, probs, confmat

def compose(*fs):
    def compose2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))
    return reduce(compose2, fs)


def time_function(f, *a, **kw):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    tic = time.time()
    f(*a, **kw)
    toc = time.time()
    return toc - tic


def unstable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__}() is unstable", category=FutureWarning, stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper


def to_np(x):
    return x.cpu().detach().numpy()


def reassign(x):
    _, col_inds = linear_sum_assignment(-x)
    return x[:, col_inds]


@unstable
def mk_masks(bias: th.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return np.where(bias.cpu() != 0)[0], np.where(bias.cpu() == 0)[0]


def compute_labels(probs):
    return np.argmax(probs, axis=-1)


# Note, if K is None, all labels are assumed to be present in at least one of the arrays
def compute_confmat(labels1, labels2, K=None):
    assert len(labels1) == len(labels2)
    assert len(labels1.shape) == len(labels2.shape) == 1
    assert labels1.dtype == labels2.dtype == np.int64

    if K is None:
        K = max(len(np.unique(labels1)), len(np.unique(labels2)))

    matrix = np.zeros((K, K))
    np.add.at(matrix, (labels1, labels2), 1)
    return matrix


def confmat_normalize(cm):
    maxes = np.maximum(np.sum(cm, axis=0), np.sum(cm, axis=1))
    return np.divide(cm, maxes, out=np.zeros_like(cm), where=maxes != 0)


def compute_confmat_naive(labels1, labels2, K=None):
    assert len(labels1) == len(labels2)
    assert len(labels1.shape) == len(labels2.shape) == 1
    assert labels1.dtype == labels2.dtype == np.int64

    if K is None:
        K = max(len(np.unique(labels1)), len(np.unique(labels2)))

    matrix = np.zeros((K, K))
    for i in range(len(labels1)):
        matrix[labels1[i], labels2[i]] += 1
    return matrix


def confmat_normalize_naive(cm):
    axis_maxes = []
    for k in range(cm.shape[0]):
        sum_row = np.sum(cm[k, :])
        sum_col = np.sum(cm[:, k])
        axis_maxes.append(max(sum_row, sum_col))
    matrix = np.divide(cm, np.array(axis_maxes), out=np.zeros_like(cm), where=axis_maxes != 0)
    return matrix


def confmat_mean(cm):
    return np.mean(np.diag(cm))


# Note, all labels are assumed to be present in both arrays
def ecdf(labels):
    assert len(labels.shape) == 1
    return np.bincount(labels) / len(labels)

def noExt(text):
    return ''.join(text.split('.')[:-1])

def parse_epoch(s: str):
  try:
    return int(noExt(s).split('_epoch_')[-1])
  except:
    return s