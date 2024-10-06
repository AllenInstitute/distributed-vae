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