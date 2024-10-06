import functools
import warnings

import torch as th
import numpy as np


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


@unstable
def mk_masks(bias: th.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return np.where(bias.cpu() != 0)[0], np.where(bias.cpu() == 0)[0]


@unstable
# TODO: add type hints that this takes one-hot encoded labels
# Note, all labels are assumed to be present in both arrays
def confusion_matrix(labels1, labels2):
    assert len(labels1) == len(labels2)
    assert len(labels1.shape) == len(labels2.shape) == 1
    K1, K2 = len(np.unique(labels1)), len(np.unique(labels2))
    matrix = np.zeros((K1, K2), dtype=int)
    np.add.at(matrix, (labels1, labels2), 1)
    return matrix
