from typing import Any, Mapping

import torch as th
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from mmidas._utils import unstable, mk_masks, to_np


@unstable
def generate(f: nn.Module, dl: DataLoader) -> Mapping[str, Any]:
    A = f.n_arm
    K = f.n_categories
    N = len(dl.dataset)
    D = f.input_dim
    D_low = f.lowD_dim
    S = f.state_dim
    B = dl.batch_size
    dev = f.device

    pruning_mask, inds_prune = mk_masks(f.fcc[0].bias)
    x_recs = np.zeros((A, N, D))
    s_means = np.zeros((A, N, S))
    s_logvars = np.zeros((A, N, S))
    cs = np.zeros((A, N, K))
    c_smps = np.zeros((A, N, K))
    x_lows = np.zeros((A, N, D_low))
    inds_x = np.zeros(N)
    losses = []
    c_dists = []
    c_l2_dists = []
    loss_recs = [[] for _ in range(A)]
    lls = [[] for _ in range(A)]
    preds = np.zeros((A, N))
    f.eval()
    with th.no_grad():
        for i, (x, i_x) in tqdm(enumerate(dl), total=len(dl)):
            x, i_x = x.to(dev), i_x.to(int)
            n_fst, n_lst = i * B, min((i + 1) * B, N)

            xs = [x for _ in range(A)]
            _x_recs, p_xs, r_xs, _x_lows, _cs, _, _c_smps, _s_means, _s_logvars, _ = f(
                xs, temp=1.0, prior_c=0, eval=True, mask=pruning_mask
            )
            _loss, _loss_recs, _, _, _c_dists, _c_l2_dists, _, _, _lls = f.loss(
                _x_recs, p_xs, r_xs, xs, _s_means, _s_logvars, _cs, _c_smps, 0
            )
            losses.append(_loss.item())
            c_dists.append(_c_dists.item())
            c_l2_dists.append(_c_l2_dists.item())
            for a, (loss_rec, ll) in enumerate(zip(_loss_recs, _lls)):
                loss_recs[a].append(loss_rec.item())
                lls[a].append(ll.item())
            outs = map(
                lambda ys: map(to_np, ys),
                zip(_s_means, _s_logvars, _cs, _c_smps, _x_lows, _x_recs),
            )
            for a, (s_mean, s_logvar, c, c_smp, x_low, x_rec) in enumerate(outs):
                s_means[a, n_fst:n_lst, :] = s_mean
                s_logvars[a, n_fst:n_lst, :] = s_logvar
                cs[a, n_fst:n_lst, :] = c
                c_smps[a, n_fst:n_lst, :] = c_smp
                x_lows[a, n_fst:n_lst, :] = x_low
                x_recs[a, n_fst:n_lst, :] = x_rec
                inds_x[n_fst:n_lst] = i_x.cpu().numpy().astype(int)
                preds[a, n_fst:n_lst] = np.argmax(c, axis=-1) + 1
    return {
        "x_recs": x_recs,
        "s_means": s_means,
        "s_logvars": s_logvars,
        "cs": cs,
        "c_smps": c_smps,
        "x_lows": x_lows,
        "inds_x": inds_x,
        "losses": losses,
        "c_dists": np.mean(np.array(c_dists)),
        "c_l2_dists": np.mean(np.array(c_l2_dists)),
        "loss_recs": np.array([np.mean(np.array(loss_recs[a])) for a in range(A)]),
        "lls": np.array([np.mean(np.array(lls[a])) for a in range(A)]),
        "inds_prune": inds_prune,
        "pruning_mask": pruning_mask,
        "preds": preds,
    }


class VAE(nn.Module): ...


class MMIDAS(nn.Module): ...


class Augmenter(nn.Module): ...


class Discriminator(nn.Module): ...


def make_augmenter(dataset): ...
