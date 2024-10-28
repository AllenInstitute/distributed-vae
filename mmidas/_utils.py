from functools import reduce, wraps
from itertools import product
import warnings
import time
from typing import List

import torch as th
from torch import nn
import numpy as np
from scipy.optimize import linear_sum_assignment

from mmidas.utils.dataloader import load_data, get_loaders
from mmidas.utils.tools import get_paths
from mmidas._evals import evals2
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
    matrix = np.divide(
        cm, np.array(axis_maxes), out=np.zeros_like(cm), where=axis_maxes != 0
    )
    return matrix


def confmat_mean(cm):
    return np.mean(np.diag(cm))


def compute_consensus_statistics(A: int, runs: List[int], epochs: int):
    from mmidas.model import load_vae

    SEED = 546

    dataset = "mouse_smartseq"
    config = get_paths("mmidas.toml", dataset)
    data = load_data(config[dataset]["data_path"] / config[dataset]["anndata_file"])
    train_loader, val_loader, all_loader = get_loaders(dataset=data["log1p"], batch_size=5000, seed=SEED)
    vaes = {r: load_vae(A, r, epochs, data["log1p"].shape[1]) for r in runs}

    loader = val_loader

    css = {}
    stds = {}
    means = {}

    l2s = {}
    stds_l2 = {}
    means_l2 = {}

    logs = {}
    stds_log = {}
    means_log = {}
    for (j, ra) in enumerate(runs):
        for rb in runs[j + 1:]:
            if ra != rb:
                ev = evals2(vaes[ra], vaes[rb], loader)
                i = 0
                for a in range(A):
                    for b in range(A):
                        avg_css = np.mean(np.diag(reassign(ev["consensus"][i])))
                        avg_l2 = np.mean(np.diag(reassign(ev["dist_l2"][i])))
                        # avg_log = np.mean(np.diag(reassign(ev["dist_log"][i])))
                        if (ra, rb) not in css:
                            css[(ra, rb)] = []
                            stds[(ra, rb)] = []
                            means[(ra, rb)] = []
                            l2s[(ra, rb)] = []
                            stds_l2[(ra, rb)] = []
                            means_l2[(ra, rb)] = []
                            logs[(ra, rb)] = []
                            stds_log[(ra, rb)] = []
                            means_log[(ra, rb)] = []
                        css[(ra, rb)].append(avg_css)
                        l2s[(ra, rb)].append(avg_l2)
                        # logs[(ra, rb)].append(avg_log)
                        i += 1
                css[(ra, rb)] = np.array(css[(ra, rb)])
                l2s[(ra, rb)] = np.array(l2s[(ra, rb)])
                # logs[(ra, rb)] = np.array(logs[(ra, rb)])

                means[(ra, rb)] = np.mean(css[(ra, rb)])
                stds[(ra, rb)] = np.std(css[(ra, rb)].flatten())
                means_l2[(ra, rb)] = np.mean(l2s[(ra, rb)])
                stds_l2[(ra, rb)] = np.std(l2s[(ra, rb)].flatten())
                # means_log[(ra, rb)] = np.mean(logs[(ra, rb)])
                # stds_log[(ra, rb)] = np.std(logs[(ra, rb)].flatten())

    for r in runs:
        ev = evals2(vaes[r], vaes[r], loader)
        i = 0
        for a in range(A):
            for b in range(A):
                if b > a:
                    avg_css = np.mean(np.diag(reassign(ev["consensus"][i])))
                    avg_l2 = np.mean(np.diag(reassign(ev["dist_l2"][i])))
                    # avg_log = np.mean(np.diag(reassign(ev["dist_log"][i])))
                    if (r, r) not in css:
                        css[(r, r)] = []
                        stds[(r, r)] = []
                        means[(r, r)] = []
                        l2s[(r, r)] = []
                        stds_l2[(r, r)] = []
                        means_l2[(r, r)] = []
                        logs[(r, r)] = []
                        stds_log[(r, r)] = []
                        means_log[(r, r)] = []
                    css[(r, r)].append(avg_css)
                    l2s[(r, r)].append(avg_l2)
                    # logs[(r, r)].append(avg_log)
                i += 1
        css[(r, r)] = np.array(css[(r, r)])
        l2s[(r, r)] = np.array(l2s[(r, r)])
        # logs[(r, r)] = np.array(logs[(r, r)])s

        means[(r, r)] = np.mean(css[(r, r)])
        stds[(r, r)] = np.std(css[(r, r)].flatten())
        means_l2[(r, r)] = np.mean(l2s[(r, r)])
        stds_l2[(r, r)] = np.std(l2s[(r, r)].flatten())
        # means_log[(r, r)] = np.mean(logs[(r, r)])
        # stds_log[(r, r)] = np.std(logs[(r, r)].flatten())

    within_run_css_xs = []
    between_run_css_xs = []
    within_run_l2s_xs = []
    between_run_l2s_xs = []
    within_run_logs_xs = []
    between_run_logs_xs = []
    for (ra, rb) in css:
        if ra == rb:
            within_run_css_xs += css[(ra, rb)].tolist()
            within_run_l2s_xs += l2s[(ra, rb)].tolist()
            # within_run_logs_xs += logs[(ra, rb)].tolist()
        else:
            between_run_css_xs += css[(ra, rb)].tolist()
            between_run_l2s_xs += l2s[(ra, rb)].tolist()
            # between_run_logs_xs += logs[(ra, rb)].tolist()



    return {
        "consensus": {
            "xs": css,
            "stds": stds,
            "means": means,
        },
        "l2": {
            "xs": l2s,
            "stds": stds_l2,
            "means": means_l2,
        },
        "log": {
            "xs": logs,
            "stds": stds_log,
            "means": means_log,
        },
        "total": {
            "within_run": {
                "css/mean": np.mean(np.array(within_run_css_xs)),
                "css/std": np.std(np.array(within_run_css_xs)),
                "l2/mean": np.mean(np.array(within_run_l2s_xs)),
                "l2/std": np.std(np.array(within_run_l2s_xs)),
                "log/mean": np.mean(np.array(within_run_logs_xs)),
                "log/std": np.std(np.array(within_run_logs_xs)),
            },
            "between_run": {
                "css/mean": np.mean(np.array(between_run_css_xs)),
                "css/std": np.std(np.array(between_run_css_xs)),
                "l2/mean": np.mean(np.array(between_run_l2s_xs)),
                "l2/std": np.std(np.array(between_run_l2s_xs)),
                "log/mean": np.mean(np.array(between_run_logs_xs)),
                "log/std": np.std(np.array(between_run_logs_xs)),
            },
        }
    }
    



# Note, all labels are assumed to be present in both arrays
def ecdf(labels):
    assert len(labels.shape) == 1
    return np.bincount(labels) / len(labels)


def noExt(text):
    return "".join(text.split(".")[:-1])


def parse_epoch(s: str):
    try:
        return int(noExt(s).split("_epoch_")[-1])
    except:
        return s


def compare_state_dicts(model1: nn.Module, model2: nn.Module, rtol=1e-5, atol=1e-8):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    if state_dict1.keys() != state_dict2.keys():
        print("The state dictionaries have different keys.")
        return False

    for key in state_dict1.keys():
        if not th.allclose(state_dict1[key], state_dict2[key], rtol=rtol, atol=atol):
            print(f"Mismatch found in layer: {key}")
            return False

    print("The state dictionaries are identical within the specified tolerance.")
    return True