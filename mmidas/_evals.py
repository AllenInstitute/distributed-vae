from typing import Any, Mapping

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def evals2(fa: nn.Module, fb: nn.Module, dl: DataLoader, eps=1e-9) -> Mapping[str, Any]:
    from mmidas.model import generate


    C = fa.n_categories
    outs_a = generate(fa, dl)
    outs_b = generate(fb, dl)

    preds_a = outs_a["preds"]
    preds_b = outs_b["preds"]
    inds_prune = outs_a["inds_prune"]

    qcas = outs_a["cs"]
    qcbs = outs_b["cs"]

    consensus = []
    consensus_min = []
    consensus_mean = []
    dist_l2 = []
    dist_log = []
    pm = []
    emp_l2 = []
    emp_log = []

    consensus_a = []
    consensus_min_a = []
    consensus_mean_a = []
    pm_a = []
    dist_l2_a = []
    dist_log_a = []
    emp_l2_a = []
    emp_log_a = []

    consensus_b = []
    consensus_min_b = []
    consensus_mean_b = []
    pm_b = []
    dist_l2_b = []
    dist_log_b = []
    emp_l2_b = []
    emp_log_b = []
    for a, pred_a in tqdm(enumerate(preds_a), total=len(preds_a)):
        for b, pred_b in enumerate(preds_b):
            _pm = np.zeros((C, C))  # performance matrix for arm a vs arm b
            _emp_l2 = np.zeros((C, C))  # empirical matrix for arm a vs arm b
            _emp_log = np.zeros((C, C))  # empirical matrix for arm a vs arm b
            for cat_a, cat_b, qca, qcb in zip(pred_a, pred_b, qcas[a], qcbs[b]):
                i_a = cat_a.astype(int) - 1
                i_b = cat_b.astype(int) - 1
                _pm[i_a, i_b] += 1
                _emp_l2[i_a, i_b] += np.sqrt((qca[i_a] - qcb[i_b]) ** 2)
                _emp_log[i_a, i_b] += 0.5 * (
                    qca[i_a] * np.log(qca[i_a] / (qcb[i_b] + eps))
                    + qcb[i_b] * np.log(qcb[i_b] / (qca[i_a] + eps))
                )
            smp_cts = []
            for c in range(C):
                smp_cts.append(max(_pm[c, :].sum(), _pm[:, c].sum()))
            smp_cts = np.array(smp_cts)
            inds_unpruned = np.where(np.isin(range(C), inds_prune) == False)[0]
            __consensus = np.divide(
                _pm, smp_cts, out=np.zeros_like(_pm), where=smp_cts != 0
            )
            _consensus = __consensus[:, inds_unpruned][inds_unpruned]
            _dist_l2 = np.divide(
                _emp_l2, smp_cts, out=np.zeros_like(_emp_l2), where=smp_cts != 0
            )[:, inds_unpruned][inds_unpruned]
            _dist_log = np.divide(
                _emp_log, smp_cts, out=np.zeros_like(_emp_log), where=smp_cts != 0
            )[:, inds_unpruned][inds_unpruned]

            consensus.append(_consensus)
            consensus_min.append(np.min(np.diag(_consensus)))
            consensus_mean.append(
                1.0 - ((np.abs(preds_a[0] - preds_b[0]) > 0.0).sum() / preds_a.shape[1])
            )
            pm.append(_pm[inds_unpruned][:, inds_unpruned])
            dist_l2.append(_dist_l2)
            dist_log.append(_dist_log)
            pm.append(_pm[inds_unpruned][:, inds_unpruned])
            emp_l2.append(_emp_l2[inds_unpruned][:, inds_unpruned])
            emp_log.append(_emp_log[inds_unpruned][:, inds_unpruned])

        for b, pred_b in enumerate(preds_a[a + 1 :]):
            _pm = np.zeros((C, C))
            _emp_l2 = np.zeros((C, C))
            _emp_log = np.zeros((C, C))
            for samp_a, samp_b, qca, qcb in zip(pred_a, pred_b, qcas[a], qcas[b]):
                i_a = samp_a.astype(int) - 1
                i_b = samp_b.astype(int) - 1
                _pm[i_a, i_b] += 1
                _emp_l2[i_a, i_b] += np.sqrt((qca[i_a] - qcb[i_b]) ** 2)
                _emp_log[i_a, i_b] += 0.5 * (
                    qca[i_a] * np.log(qca[i_a] / (qcb[i_b] + eps))
                    + qcb[i_b] * np.log(qcb[i_b] / (qca[i_a] + eps))
                )

            smp_cts = []
            for c in range(C):
                smp_cts.append(max(_pm[c].sum(), _pm[:, c].sum()))
            smp_cts = np.array(smp_cts)

            inds_unpruned = np.where(np.isin(range(C), inds_prune) == False)[0]
            _consensus = np.divide(
                _pm, smp_cts, out=np.zeros_like(_pm), where=smp_cts != 0
            )[:, inds_unpruned][inds_unpruned]
            _dist_l2 = np.divide(
                _emp_l2, smp_cts, out=np.zeros_like(_emp_l2), where=smp_cts != 0
            )[:, inds_unpruned][inds_unpruned]
            _dist_log = np.divide(
                _emp_log, smp_cts, out=np.zeros_like(_emp_log), where=smp_cts != 0
            )[:, inds_unpruned][inds_unpruned]

            consensus_a.append(_consensus)
            consensus_min_a.append(np.min(np.diag(_consensus)))
            consensus_mean_a.append(
                1.0 - ((np.abs(preds_a[0] - preds_a[1]) > 0.0).sum() / preds_a.shape[1])
            )
            pm_a.append(_pm[inds_unpruned][:, inds_unpruned])
            dist_l2_a.append(_dist_l2)
            dist_log_a.append(_dist_log)
            emp_l2_a.append(_emp_l2[inds_unpruned][:, inds_unpruned])
            emp_log_a.append(_emp_log[inds_unpruned][:, inds_unpruned])

    for a, pred_a in tqdm(enumerate(preds_b), total=len(preds_b)):
        for b, pred_b in enumerate(preds_b[a + 1 :]):
            _pm = np.zeros((C, C))
            _emp_l2 = np.zeros((C, C))
            _emp_log = np.zeros((C, C))
            for samp_a, samp_b, qca, qcb in zip(pred_a, pred_b, qcbs[a], qcbs[b]):
                i_a = samp_a.astype(int) - 1
                i_b = samp_b.astype(int) - 1
                _pm[i_a, i_b] += 1
                _emp_l2[i_a, i_b] += np.sqrt((qca[i_a] - qcb[i_b]) ** 2)
                _emp_log[i_a, i_b] += 0.5 * (
                    qca[i_a] * np.log(qca[i_a] / (qcb[i_b] + eps))
                    + qcb[i_b] * np.log(qcb[i_b] / (qca[i_a] + eps))
                )

            smp_cts = []
            for c in range(C):
                smp_cts.append(max(_pm[c].sum(), _pm[:, c].sum()))
            smp_cts = np.array(smp_cts)

            inds_unpruned = np.where(np.isin(range(C), inds_prune) == False)[0]
            _consensus = np.divide(
                _pm, smp_cts, out=np.zeros_like(_pm), where=smp_cts != 0
            )[:, inds_unpruned][inds_unpruned]
            _dist_l2 = np.divide(
                _emp_l2, smp_cts, out=np.zeros_like(_emp_l2), where=smp_cts != 0
            )[:, inds_unpruned][inds_unpruned]
            _dist_log = np.divide(
                _emp_log, smp_cts, out=np.zeros_like(_emp_log), where=smp_cts != 0
            )[:, inds_unpruned][inds_unpruned]

            consensus_b.append(_consensus)
            consensus_min_b.append(np.min(np.diag(_consensus)))
            consensus_mean_b.append(
                1.0 - ((np.abs(preds_b[0] - preds_b[1]) > 0.0).sum() / preds_b.shape[1])
            )
            pm_b.append(_pm[inds_unpruned][:, inds_unpruned])
            dist_l2_b.append(_dist_l2)
            dist_log_b.append(_dist_log)
            emp_l2_b.append(_emp_l2[inds_unpruned][:, inds_unpruned])
            emp_log_b.append(_emp_log[inds_unpruned][:, inds_unpruned])

    return {
        "consensus": consensus,
        "consensus_min": consensus_min,
        "consensus_mean": consensus_mean,
        "pm": pm,
        "consensus_a": consensus_a,
        "consensus_min_a": consensus_min_a,
        "consensus_mean_a": consensus_mean_a,
        "pm_a": pm_a,
        "consensus_b": consensus_b,
        "consensus_min_b": consensus_min_b,
        "consensus_mean_b": consensus_mean_b,
        "pm_b": pm_b,
        "inds_unpruned": inds_unpruned,
        "cs_a": outs_a["cs"],
        "cs_b": outs_b["cs"],
        "dist_l2": dist_l2,
        "dist_log": dist_log,
        "emp_l2": emp_l2,
        "emp_log": emp_log,
        "dist_l2_a": dist_l2_a,
        "dist_log_a": dist_log_a,
        "emp_l2_a": emp_l2_a,
        "emp_log_a": emp_log_a,
        "dist_l2_b": dist_l2_b,
        "dist_log_b": dist_log_b,
        "emp_l2_b": emp_l2_b,
    }
