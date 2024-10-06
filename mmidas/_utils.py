import functools
import warnings

import torch as th
import numpy as np
from scipy.optimize import linear_sum_assignment

# types: labels, probs, confmat

def unstable(func):
    @functools.wraps(func)
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


# Note, all labels are assumed to be present in at least one of the arrays
def compute_confmat(labels1, labels2, K=None):
    assert len(labels1) == len(labels2)
    assert len(labels1.shape) == len(labels2.shape) == 1
    assert labels1.dtype == labels2.dtype == np.int64
    if K is None:
        K = max(len(np.unique(labels1)), len(np.unique(labels2)))
    matrix = np.zeros((K, K), dtype=int)
    np.add.at(matrix, (labels1, labels2), 1)
    return matrix


def confmat_normalize(cm):
    # algebra: this is an evaluation/reduction operation
    maxes = np.maximum(np.sum(cm, axis=0), np.sum(cm, axis=1))
    return np.divide(cm, maxes)


def confmat_mean(cm):
    return np.mean(np.diag(cm))


# Note, all labels are assumed to be present in both arrays
def ecdf(labels):
    assert len(labels.shape) == 1
    return np.bincount(labels) / len(labels)
