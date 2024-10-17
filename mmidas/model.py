import glob
from typing import Any, Mapping, Literal

import torch as th
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from mmidas.nn_model import mixVAE_model, mk_vae
from mmidas.utils.tools import get_paths
from mmidas._utils import unstable, mk_masks, to_np, parse_epoch


# TODO
def clr(prob, eps=1e-8):
    assert th.all((0 <= prob) & (prob <= 1)) and th.sum(prob, dim=-1) == 1.0


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


Run = str | int
Config = Mapping[str, Any]
TrainedModelFile = str
Summary = Mapping[Any, Any]
Arm = int
Arms = int

Consensus = float
MI = float

TrainedModel = nn.Module

Bias = th.Tensor
PruningMask = th.Tensor | np.ndarray
PruningIndex = th.Tensor

MixVAE = mixVAE_model

MouseSmartSeq = Literal["mouse_smartseq"]
Mouse10x = Literal["TODO"]
SeattleAlzheimer = Literal["TODO"]

Dataset = MouseSmartSeq | Mouse10x | SeattleAlzheimer

def load_vae(arms: int, run: int, epochs: int, input_dim: int) -> MixVAE:
    r = mk_run(arms, run, epochs)
    vae = mk_vae(**_mk_vae_cfg(arms, input_dim))
    print(get_weights(r, MouseSmartSeq))
    load_weights(vae, get_weights(r, MouseSmartSeq))
    return vae


def _mk_vae_cfg(A: int, input_dim: int) -> Mapping[str, Any]:
    return {
        "C": 92,
        "state_dim": 2,
        "input_dim": input_dim,
        "device": "cuda",
        "A": A,
        "latent_dim": 10,
    }


def unwrap_literal(x: Dataset) -> str:
    return x.__args__[0]


def mk_run(arms: int, run: int, epochs: int = 500000) -> Run:
    s = f"K92_S2_AUGTrue_LR0.001_A{arms}_B5000_E{epochs}_Ep0_RUN{run}"
    print("run:", s)
    return s


def mk_config(r: str, d: Dataset) -> Config:
    config = get_paths("mmidas.toml", unwrap_literal(d))
    config["mouse_smartseq"]["trained_model"] = r
    return config


def get_weights(r: Run, d: Dataset) -> TrainedModelFile:
    c: Config = mk_config(r, d)
    saving_folder = c["paths"]["main_dir"] / c[unwrap_literal(d)]["saving_path"]
    trained_model_folder = c[unwrap_literal(d)]["trained_model"]
    saving_folder = str(saving_folder / trained_model_folder)
    trained_models = glob.glob(saving_folder + "/model/cpl_mixVAE_model_before**")
    if len(trained_models) == 0:
        trained_models = glob.glob(saving_folder + "/model/cpl_mixVAE_model_epoch**")
        
    if len(trained_models) == 1:
        found = trained_models[0]
    elif len(trained_models) > 1:
        found = max(trained_models, key=parse_epoch)
    else:
        raise FileNotFoundError("no trained model found")
    print("loading:", found)
    return found


def view_weights(arms: int, run: int) -> None:
    return th.load(get_weights(mk_run(arms, run), MouseSmartSeq), map_location="cpu")[
        "model_state_dict"
    ]


def load_weights(m: nn.Module, f: str) -> None:
    m.load_state_dict(th.load(f, map_location="cpu")["model_state_dict"])


class VAE(nn.Module): ...


class MMIDAS(nn.Module): ...


class Augmenter(nn.Module): ...


class Discriminator(nn.Module): ...


def make_augmenter(dataset): ...
