import os
import pickle
import random
import time
from functools import reduce
from itertools import cycle, repeat
from dataclasses import dataclass
from typing import Optional, Literal, assert_never, Sequence, Iterable, Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
from torch import nn, cuda
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm, trange
import wandb

from .augmentation.udagan import *
from .nn_model import mixVAE_model, VAEConfig
from .utils.data_tools import split_data_Kfold

from mmidas._utils import to_np, compute_labels, compute_confmat, confmat_mean, confmat_normalize


def bytes_to_mb(x):
    return x / 1e6


def is_master(rank):
    return rank == 0 or rank == "mps" or rank == "cpu"


def compose(*fs):
    def compose2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))
    return reduce(compose2, fs)


def mk_pbar(seq: Sequence, r, *fs) -> Sequence:
    return (lambda x: tqdm(x, total=len(x)) if is_master(r) else x)(compose(*fs)(seq))  # type: ignore


def is_parallel(world_size):
    return world_size > 1


def print_train_loss(
    epoch,
    train_loss,
    train_recon0,
    train_recon1,
    train_loss_joint,
    train_entropy,
    train_distance,
    time,
    rank,
):
    if is_master(rank):
        print(
            "====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {:.4f}, Rec_arm_2: {:.4f}, Joint Loss: {:.4f}, Entropy: {:.4f}, Distance: {:.4f}, Elapsed Time:{:.2f}".format(
                epoch,
                train_loss,
                train_recon0,
                train_recon1,
                train_loss_joint,
                train_entropy,
                train_distance,
                time,
            )
        )


def print_val_loss(val_loss, val_loss_rec, rank):
    if is_master(rank):
        print(
            "====> Validation Total Loss: {:.4f}, Rec. Loss: {:.4f}".format(
                val_loss, val_loss_rec
            )
        )


def unwrap[T](x: Optional[T]) -> T:
    if x is None:
        raise ValueError("error: expected non-None value")
    return x


def get_device(device: Optional[str | int] = None) -> th.device:
    match device:
        case "cpu" | "mps" as d:
            print(f"using {d}")
            return th.device(d)
        case "cuda" as d:
            print(f"using {d}: {cuda.get_device_name(d)}")
            return th.device(d)
        case int(d):
            cuda.set_device(d)
            return get_device("cuda")
        case None:
            print("device not found")
            return get_device("cpu")
        case _:
            assert_never(device)


def mk_augmenter(
    pretrained: str, load_weights: bool
) -> tuple[Mapping[Any, Any], Mapping[Any, Any], nn.Module]:
    aug_model = th.load(pretrained, map_location="cpu")
    aug_param = aug_model["parameters"]
    if load_weights:
        print("loading augmenter weights")
        netA = Augmenter_smartseq(
            noise_dim=aug_param["num_n"],
            latent_dim=aug_param["num_z"],
            input_dim=aug_param["n_features"],
        )
        netA.load_state_dict(aug_model["netA"])
        return aug_model, aug_param, netA
    else:
        print("warning: not loading augmenter weights")
        netA = Augmenter(
            noise_dim=aug_param["num_n"],
            latent_dim=aug_param["num_z"],
            input_dim=aug_param["n_features"],
        )
        return aug_model, aug_param, netA


class cpl_mixVAE:
    def __init__(
        self,
        saving_folder="",
        aug_file="",
        device = None,
        eps=1e-8,
        save_flag=True,
        load_weights=True,
    ):
        """
        Initialized the cpl_mixVAE class.

        input args:
            saving_folder: a string that indicates the folder to save the model(s) and file(s).
            aug_file: a string that indicates the file of the pre-trained augmenter.
            device: computing device, either 'cpu' or 'cuda'.
            eps: a small constant value to fix computation overflow.
            save_flag: a boolean variable, if True, the model is saved.
        """

        self.eps = eps
        self.save = save_flag
        self.folder = saving_folder
        self.aug_file = aug_file
        self.device = device
        self.models: list[dict[str, nn.Module | Optimizer]] = []

        self.device = get_device(device)

        if aug_file:
            self.aug_model, self.aug_param, netA = mk_augmenter(aug_file, load_weights)
            self.netA = netA.to(self.device).eval()
        else:
            self.aug_model, self.aug_param, self.netA = None, None, None


    def init_model(
        self,
        n_categories,
        state_dim,
        input_dim,
        fc_dim=100,
        lowD_dim=10,
        x_drop=0.5,
        s_drop=0.2,
        lr=0.001,
        lam=1,
        lam_pc=1,
        n_arm=2,
        temp=1.0,
        tau=0.005,
        beta=1.0,
        hard=False,
        variational=True,
        ref_prior=False,
        trained_model="",
        n_pr=0,
        momentum=0.01,
        mode="MSE",
    ):
        """
        Initialized the deep mixture model and its optimizer.

        input args:
            n_categories: number of categories of the latent variables.
            state_dim: dimension of the state variable.
            input_dim: input dimension (size of the input layer).
            fc_dim: dimension of the hidden layer.
            lowD_dim: dimension of the latent representation.
            x_drop: dropout probability at the first (input) layer.
            s_drop: dropout probability of the state variable.
            lr: the learning rate of the optimizer, here Adam.
            lam: coupling factor in the cpl-mixVAE model.
            lam_pc: coupling factor for the prior categorical variable.
            n_arm: int value that indicates number of arms.
            temp: temperature of sampling
            tau: temperature of the softmax layers, usually equals to 1/n_categories (0 < tau <= 1).
            beta: regularizer for the KL divergence term.
            hard: a boolean variable, True uses one-hot method that is used in Gumbel-softmax, and False uses the Gumbel-softmax function.
            variational: a boolean variable for variational mode, False mode does not use sampling.
            ref_prior: a boolean variable, True uses the reference prior for the categorical variable.
            trained_model: a pre-trained model, in case you want to initialized the network with a pre-trained network.
            n_pr: number of pruned categories, only if you want to initialize the network with a pre-trained network.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
            mode: the loss function, either 'MSE' or 'ZINB'.
        """
        self.lowD_dim = lowD_dim
        self.n_categories = n_categories
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.temp = temp
        self.n_arm = n_arm
        self.fc_dim = fc_dim
        self.ref_prior = ref_prior
        self.model = mixVAE_model(
            input_dim=self.input_dim,
            fc_dim=fc_dim,
            n_categories=self.n_categories,
            state_dim=self.state_dim,
            lowD_dim=lowD_dim,
            x_drop=x_drop,
            s_drop=s_drop,
            n_arm=self.n_arm,
            lam=lam,
            lam_pc=lam_pc,
            tau=tau,
            beta=beta,
            hard=hard,
            variational=variational,
            device=self.device,
            eps=self.eps,
            ref_prior=ref_prior,
            momentum=momentum,
            loss_mode=mode,
        )

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if len(trained_model) > 0:
            print("Load the pre-trained model")
            # if you wish to load another model for evaluation
            loaded_file = torch.load(trained_model, map_location="cpu")
            self.model.load_state_dict(loaded_file["model_state_dict"])
            self.optimizer.load_state_dict(loaded_file["optimizer_state_dict"])
            self.init = False
            self.n_pr = n_pr
        else:
            self.init = True
            self.n_pr = 0

    def append(self, c: VAEConfig):
        model = mixVAE_model(
            input_dim=c.input_dim,
            fc_dim=c.fc_dim,
            n_categories=c.n_categories,
            state_dim=c.state_dim,
            lowD_dim=c.lowD_dim,
            x_drop=c.x_drop,
            s_drop=c.s_drop,
            n_arm=c.n_arm,
            lam=c.lam,
            lam_pc=c.lam_pc,
            tau=c.tau,
            beta=c.beta,
            hard=c.hard,
            variational=c.variational,
            device=self.device,
            eps=self.eps,
            ref_prior=c.ref_prior,
            momentum=c.momentum,
            loss_mode=c.mode,
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        if c.trained_model:
            loaded_file = th.load(c.trained_model, map_location="cpu")
            model.load_state_dict(loaded_file["model_state_dict"])
            optimizer.load_state_dict(loaded_file["optimizer_state_dict"])
        self.models.append({"model": model, "opt": optimizer})

    def load_model(self, trained_model):
        loaded_file = torch.load(trained_model, map_location="cpu")
        self.model.load_state_dict(loaded_file["model_state_dict"])

        self.current_time = time.strftime("%Y-%m-%d-%H-%M-%S")

    # TODO: add stopping criteria for training
    def train(
        self,
        train_loader,
        test_loader,
        n_epoch,
        n_epoch_p,
        c_p=0,
        c_onehot=0,
        min_con=0.5,
        max_prun_it=0,
        rank=None,
        run=None,
        ws=1,
        good_enuf_consensus=0.8
    ):
        """
        run the training of the cpl-mixVAE with the pre-defined parameters/settings
        pcikle used for saving the file

        input args
            train_loader: train dataloader.
            test_loader: test dataloader.
            n_epoch: number of training epoch, without pruning.
            n_epoch_p: number of training epoch, with pruning.
            c_p: the prior categorical variable, only if ref_prior is True.
            c_onehot: the one-hot representation of the prior categorical variable, only if ref_prior is True.
            min_con: minimum value of consensus among pair of arms.
            max_prun_it: maximum number of pruning iterations.
            mode: the loss function, either 'MSE' or 'ZINB'.

        return
            data_file_id: the output dictionary.
        """
        if rank is None:
            rank = self.device

        A = self.n_arm
        C = self.n_categories
        E = n_epoch
        D = self.input_dim
        D_low = self.lowD_dim
        B = train_loader.batch_size
        B_val = test_loader.batch_size
        Bs = len(train_loader)
        Bs_val = len(test_loader)
        S = self.state_dim
        N = len(train_loader.dataset)

        self.current_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        losses = []
        loss_joints = []
        loss_recs = [[] for _ in range(A)]
        c_ents = []
        c_l2_dists = []
        c_dists = []
        consensuss = []

        validation_loss = np.zeros(E)
        validation_rec_loss = np.zeros(E)
        bias_mask = th.ones(C)
        weight_mask = th.ones((C, D_low))
        fc_mu = th.ones((S, C + D_low))
        fc_sigma = th.ones((S, C + D_low))
        f6_mask = th.ones((D_low, S + C))

        bias_mask = bias_mask.to(rank)
        weight_mask = weight_mask.to(rank)
        fc_mu = fc_mu.to(rank)
        fc_sigma = fc_sigma.to(rank)
        f6_mask = f6_mask.to(rank)

        if self.init:
            print("training started")
            epoch_times = []
            for e in trange(E):
                loss = th.zeros(2, device=rank)
                loss_joint = th.zeros(1, device=rank)
                loss_rec = th.zeros(A, device=rank)
                c_l2_dist = th.zeros(1, device=rank)
                c_dist = th.zeros(1, device=rank)
                c_ent = th.zeros(1, device=rank)
                t0 = time.time()
                cs_train = [[] for _ in range(A)]

                probs = [[] for _ in range(A)]

                self.model.train()
                for (x, n) in train_loader:
                    x = x.to(rank)
                    n = n.to(int)

                    tt = time.time()

                    with th.no_grad():
                        if self.aug_file:
                            xs = self.netA(x.expand(A, -1, -1), True, 0.1)[1]
                        else:
                            xs = x.expand(A, -1, -1)

                    if self.ref_prior:
                        c_bin = th.tensor(c_onehot[n, :], dtype=th.float, device=rank)
                        prior_c = th.tensor(c_p[n, :], dtype=th.float, device=rank)
                    else:
                        c_bin = 0.0
                        prior_c = 0.0

                    self.optimizer.zero_grad()
                    x_recs, _, _, _, cs, _, c_smps, s_means, s_logvars, _ = self.model(
                        xs, self.temp, prior_c
                    )
                    for a in range(A):
                        cs_train[a].append(
                            cs[a]
                            .cpu()
                            .view(cs[a].size()[0], C)
                            .argmax(dim=1)
                            .detach()
                            .numpy()
                        )
                    (
                        _loss,
                        _loss_rec,
                        _loss_joint,
                        _c_ent,
                        _c_dist,
                        _c_l2_dist,
                        _,
                        _,
                        _,
                    ) = self.model.loss(
                        x_recs, [], [], xs, s_means, s_logvars, cs, c_smps, c_bin
                    )
                    mem: float = bytes_to_mb(th.cuda.memory_allocated())
                    _loss.backward()
                    self.optimizer.step()

                    loss[0] += _loss.item()
                    loss[1] += 1
                    loss_joint += _loss_joint
                    c_l2_dist += _c_l2_dist
                    c_dist += _c_dist
                    c_ent += _c_ent
                    loss_rec += _loss_rec / D
                    for a in range(A):
                        probs[a].append(to_np(cs[a]))

                if ws > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_rec, op=dist.ReduceOp.SUM)
                    dist.all_reduce(c_dist, op=dist.ReduceOp.SUM)

                losses.append(loss[0].item() / loss[1].item())
                loss_joints.append(loss_joint.item() / Bs)
                c_ents.append(c_ent.item() / Bs)
                c_l2_dists.append(c_l2_dist.item() / Bs)
                c_dists.append(c_dist.item() / loss[1].item())

                for a in range(A):
                    loss_recs[a].append(loss_rec[a].item() / loss[1].item())

                labels = [np.ravel(compute_labels(np.array(probs[a]))) for a in range(A)]
                consensus = []
                for a in range(A):
                    for b in range(a + 1, A):
                        consensus.append(confmat_mean(confmat_normalize(compute_confmat(labels[a], labels[b], C))))
                consensuss.append(np.mean(np.array(consensus)))

                _time = time.time() - t0
                print(
                    f"epoch {e} | loss: {losses[-1]:.2f} | rec: {loss_recs[0][-1]:.2f} | joint: {loss_joints[-1]} | entropy: {c_ents[-1]:.2f} | distance: {c_dists[-1]:.2f} | l2 distance: {c_l2_dists[-1]:.2f} | consensus: {consensuss[-1]:.2f} | time {_time:.2f} | mem {mem:.2f} | ",
                    end="",
                )

                if run:
                    run.log(
                        {
                            "train/total-loss": losses[-1],
                            "train/joint-loss": loss_joints[-1],
                            "train/negative-joint-entropy": c_ents[-1],
                            "train/simplex-distance": c_dists[-1],
                            "train/l2-distance": c_l2_dists[-1],
                            "train/time": _time,
                            "train/mem": mem,
                            "train/consensus": consensuss[-1],
                            **dict(
                                map(
                                    lambda a: (f"train/rec-loss{a}", loss_recs[a][-1]),
                                    range(A),
                                )
                            ),
                        }
                    )

                # validation
                self.model.eval()
                with th.no_grad():
                    val_loss = 0.0
                    val_loss_rec = 0.0
                    if B_val > 1:
                        for (
                            batch_indx,
                            (x, n),
                        ) in enumerate(test_loader):  # batch index, (data, data index)
                            x = x.to(rank)
                            n = n.to(int)

                            xs = [x for _ in range(A)]

                            if self.ref_prior:
                                c_bin = th.tensor(
                                    c_onehot[n, :], dtype=th.float, device=rank
                                )
                                prior_c = th.tensor(
                                    c_p[n, :], dtype=th.float, device=rank
                                )
                            else:
                                c_bin = 0.0
                                prior_c = 0.0

                            (
                                x_recs,
                                p_x,
                                r_x,
                                _,
                                cs,
                                _,
                                c_smps,
                                s_means,
                                s_logvars,
                                _,
                            ) = self.model(
                                x=xs, temp=self.temp, prior_c=prior_c, eval=True
                            )
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = (
                                self.model.loss(
                                    x_recs,
                                    p_x,
                                    r_x,
                                    xs,
                                    s_means,
                                    s_logvars,
                                    cs,
                                    c_smps,
                                    c_bin,
                                )
                            )
                            val_loss += loss.data.item()
                            for a in range(A):
                                val_loss_rec += loss_rec[a].item() / D
                    else:
                        batch_indx = 0
                        x, n = test_loader.dataset.tensors
                        x = x.to(rank)
                        n = n.to(int)

                        xs = [x for _ in range(A)]

                        if self.ref_prior:
                            c_bin = th.tensor(
                                c_onehot[n, :], dtype=th.float, device=rank
                            )
                            prior_c = th.tensor(c_p[n, :], dtype=th.float, device=rank)
                        else:
                            c_bin = 0.0
                            prior_c = 0.0

                        x_recs, p_x, r_x, _, cs, _, c_smps, s_means, s_logvars, _ = (
                            self.model(x=xs, temp=self.temp, prior_c=prior_c, eval=True)
                        )
                        loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(
                            x_recs, p_x, r_x, xs, s_means, s_logvars, cs, c_smps, c_bin
                        )
                        val_loss = loss.item()
                        for a in range(A):
                            val_loss_rec += loss_rec[a].item() / D

                validation_rec_loss[e] = val_loss_rec / Bs_val / A
                validation_loss[e] = val_loss / Bs_val
                print(
                    f"val-loss {validation_loss[e]:.2f} | rec-loss {validation_rec_loss[e]:.2f}"
                )
                if run:
                    run.log(
                        {
                            "val/total-loss": validation_loss[e],
                            "val/rec-loss": validation_rec_loss[e],
                        }
                    )

                if self.save and (e > 0) and (e % 5 == 0):
                    trained_model = (
                        self.folder + f"/model/cpl_mixVAE_model_epoch_{e}.pth"
                    )
                    print(f"saving model to: {trained_model}")
                    th.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        trained_model,
                    )

                    predicted_label = np.zeros((A, len(cs_train[0] * B)))
                    for a in range(A):
                        predicted_label[a] = np.concatenate(cs_train[a])

                    # confusion matrix code
                    c_agreement = []
                    for a in range(A):
                        pred_a = predicted_label[a, :]
                        for b in range(a + 1, A):
                            pred_b = predicted_label[b, :]
                            armA_vs_armB = np.zeros((C, C))

                            for samp in range(pred_a.shape[0]):
                                armA_vs_armB[
                                    pred_a[samp].astype(int), pred_b[samp].astype(int)
                                ] += 1

                            num_samp_arm = []
                            for ij in range(C):
                                sum_row = armA_vs_armB[ij, :].sum()
                                sum_column = armA_vs_armB[:, ij].sum()
                                num_samp_arm.append(max(sum_row, sum_column))

                            armA_vs_armB = np.divide(
                                armA_vs_armB,
                                np.array(num_samp_arm),
                                out=np.zeros_like(armA_vs_armB),
                                where=np.array(num_samp_arm) != 0,
                            )
                            c_agreement.append(np.diag(armA_vs_armB))
                            ind_sort = np.argsort(c_agreement[-1])
                            plt.figure()
                            plt.imshow(
                                armA_vs_armB[:, ind_sort[::-1]][ind_sort[::-1]],
                                cmap="binary",
                            )
                            plt.colorbar()
                            plt.xlabel("arm_" + str(a), fontsize=20)
                            plt.xticks(range(C), range(C))
                            plt.yticks(range(C), range(C))
                            plt.ylabel("arm_" + str(b), fontsize=20)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title(f"Epoch {e} |c|=" + str(C), fontsize=20)
                            plt.savefig(
                                self.folder
                                + "/consensus_arm_"
                                + str(a)
                                + "_arm_"
                                + str(a)
                                + "_epoch_"
                                + str(e)
                                + ".png",
                                dpi=600,
                            )
                            plt.close("all")
                if consensuss[-1] >= good_enuf_consensus:
                    break

                epoch_times.append(time.time() - t0)

            print("epoch time:", np.mean(epoch_times))

            def save_loss_plot(loss_data, label, filename):
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch), loss_data, label=label)
                ax.set_xlabel("# epoch", fontsize=16)
                ax.set_ylabel("loss value", fontsize=16)
                ax.set_title(
                    f"{label} loss of the cpl-mixVAE for K={C} and S={self.state_dim}"
                )
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.legend()
                ax.figure.savefig(
                    self.folder + f"/model/{filename}_A{A}_{C}_{self.current_time}.png"
                )
                plt.close()

            if self.save and n_epoch > 0:
                # Save train loss plot
                save_loss_plot(losses, "Training", "train_loss_curve")

                # Save validation loss plot
                save_loss_plot(validation_loss, "Validation", "validation_loss_curve")

                trained_model = (
                    self.folder
                    + f"/model/cpl_mixVAE_model_before_pruning_A{A}_"
                    + self.current_time
                    + ".pth"
                )
                print(f"saving model to: {trained_model}")
                th.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    trained_model,
                )
                bias = self.model.fcc[0].bias.detach().cpu().numpy()
                mask = range(len(bias))
                prune_indx = []
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(E), losses, label="Training")
                ax.plot(range(E), validation_loss, label="Validation")
                ax.set_xlabel("# epoch", fontsize=16)
                ax.set_ylabel("loss value", fontsize=16)
                ax.set_title(
                    "Learning curve of the cpl-mixVAE for K="
                    + str(C)
                    + " and S="
                    + str(S)
                )
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.legend()
                ax.figure.savefig(
                    self.folder
                    + f"/model/learning_curve_before_pruning_K_A{A}_"
                    + str(C)
                    + "_"
                    + self.current_time
                    + ".png"
                )
                plt.close("all")

        if n_epoch_p > 0:
            # initialized pruning parameters of the layer of the discrete variable
            bias = self.model.fcc[0].bias.detach().cpu().numpy()
            pruning_mask = np.where(bias != 0.0)[0]
            prune_indx = np.where(bias == 0.0)[0]
            stop_prune = False
        else:
            stop_prune = True

        pr = self.n_pr
        ind = []
        stop_prune = True
        print("warning: stopping pruning")
        while not stop_prune:
            predicted_label = np.zeros((self.n_arm, len(train_loader.dataset)))

            # Assessment over all dataset
            self.model.eval()
            with th.no_grad():
                for i, (data, d_idx) in enumerate(train_loader):
                    data = data.to(self.device)
                    d_idx = d_idx.to(int)
                    trans_data = []
                    for arm in range(self.n_arm):
                        trans_data.append(data)

                    if self.ref_prior:
                        c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                        prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                    else:
                        c_bin = 0.0
                        prior_c = 0.0

                    (
                        recon,
                        p_x,
                        r_x,
                        x_low,
                        z_category,
                        state,
                        z_smp,
                        mu,
                        log_sigma,
                        _,
                    ) = self.model(
                        trans_data, self.temp, prior_c, mask=pruning_mask, eval=True
                    )

                    for a in range(A):
                        z_encoder = (
                            z_category[a]
                            .cpu()
                            .data.view(z_category[a].size()[0], C)
                            .detach()
                            .numpy()
                        )
                        predicted_label[a, i * B : min((i + 1) * B, N)] = np.argmax(
                            z_encoder, axis=1
                        )

            c_agreement = []
            for arm_a in range(self.n_arm):
                pred_a = predicted_label[arm_a, :]
                for arm_b in range(arm_a + 1, self.n_arm):
                    pred_b = predicted_label[arm_b, :]
                    armA_vs_armB = np.zeros((C, C))

                    for samp in range(pred_a.shape[0]):
                        armA_vs_armB[
                            pred_a[samp].astype(int), pred_b[samp].astype(int)
                        ] += 1

                    num_samp_arm = []
                    for ij in range(C):
                        sum_row = armA_vs_armB[ij, :].sum()
                        sum_column = armA_vs_armB[:, ij].sum()
                        num_samp_arm.append(max(sum_row, sum_column))

                    armA_vs_armB = np.divide(
                        armA_vs_armB,
                        np.array(num_samp_arm),
                        out=np.zeros_like(armA_vs_armB),
                        where=np.array(num_samp_arm) != 0,
                    )
                    c_agreement.append(np.diag(armA_vs_armB))
                    ind_sort = np.argsort(c_agreement[-1])
                    plt.figure()
                    plt.imshow(
                        armA_vs_armB[:, ind_sort[::-1]][ind_sort[::-1]], cmap="binary"
                    )
                    plt.colorbar()
                    plt.xlabel("arm_" + str(arm_a), fontsize=20)
                    plt.xticks(range(C), range(C))
                    plt.yticks(range(C), range(C))
                    plt.ylabel("arm_" + str(arm_b), fontsize=20)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title("|c|=" + str(C), fontsize=20)
                    plt.savefig(
                        self.folder
                        + "/consensus_"
                        + str(pr)
                        + "_arm_"
                        + str(arm_a)
                        + "_arm_"
                        + str(arm_b)
                        + ".png",
                        dpi=600,
                    )
                    plt.close("all")

            c_agreement = np.mean(c_agreement, axis=0)
            agreement = c_agreement[pruning_mask]
            if (np.min(agreement) <= min_con) and pr < max_prun_it:
                if pr > 0:
                    ind_min = pruning_mask[np.argmin(agreement)]
                    ind_min = np.array([ind_min])
                    ind = np.concatenate((ind, ind_min))
                else:
                    ind_min = pruning_mask[np.argmin(agreement)]
                    if len(prune_indx) > 0:
                        ind_min = np.array([ind_min])
                        ind = np.concatenate((prune_indx, ind_min))
                    else:
                        ind.append(ind_min)
                    ind = np.array(ind)

                ind = ind.astype(int)
                bias_mask[ind] = 0.0
                weight_mask[ind, :] = 0.0
                fc_mu[:, self.lowD_dim + ind] = 0.0
                fc_sigma[:, self.lowD_dim + ind] = 0.0
                f6_mask[:, ind] = 0.0
                stop_prune = False
            else:
                print("No more pruning!")
                stop_prune = True

            print("warning: disabled pruning")
            stop_prune = True
            if not stop_prune:
                print("Continue training with pruning ...")
                print(f"Pruned categories: {ind}")
                bias = bias_mask.detach().cpu().numpy()
                pruning_mask = np.where(bias != 0.0)[0]
                train_loss = np.zeros(n_epoch_p)
                validation_rec_loss = np.zeros(n_epoch_p)
                total_val_loss = np.zeros(n_epoch_p)
                train_loss_joint = np.zeros(n_epoch_p)
                train_entropy = np.zeros(n_epoch_p)
                train_distance = np.zeros(n_epoch_p)
                train_minVar = np.zeros(n_epoch_p)
                train_log_distance = np.zeros(n_epoch_p)
                train_recon = np.zeros((self.n_arm, n_epoch_p))
                train_loss_KL = np.zeros((self.n_arm, self.n_categories, n_epoch_p))

                for a in range(A):
                    prune.custom_from_mask(
                        self.model.fcc[a], "weight", mask=weight_mask
                    )
                    prune.custom_from_mask(self.model.fcc[a], "bias", mask=bias_mask)
                    prune.custom_from_mask(self.model.fc_mu[a], "weight", mask=fc_mu)
                    prune.custom_from_mask(
                        self.model.fc_sigma[a], "weight", mask=fc_sigma
                    )
                    prune.custom_from_mask(self.model.fc6[a], "weight", mask=f6_mask)

                for epoch in trange(n_epoch_p):
                    # training
                    train_loss_val = 0
                    train_jointloss_val = 0
                    train_dqz = 0
                    log_dqz = 0
                    entr = 0
                    var_min = 0
                    t0 = time.time()
                    train_loss_rec = np.zeros(self.n_arm)
                    train_KLD_cont = np.zeros((self.n_arm, self.n_categories))
                    ti = np.zeros(len(train_loader))
                    self.model.train()
                    # training
                    for (
                        batch_indx,
                        (data, d_idx),
                    ) in enumerate(train_loader):
                        # for data in train_loader:
                        data = data.to(self.device)
                        d_idx = d_idx.to(int)
                        data_bin = 0.0 * data
                        data_bin[data > 0.0] = 1.0
                        trans_data = []
                        origin_data = []
                        trans_data.append(data)
                        tt = time.time()
                        w_param, bias_param, activ_param = 0, 0, 0
                        # parallelize
                        for arm in range(A - 1):
                            if self.aug_file:
                                noise = torch.randn(
                                    batch_size, self.aug_param["num_n"]
                                ).to(self.device)
                                _, gen_data = self.netA(data, noise, True, self.device)
                                # if self.aug_param['n_zim'] > 1:
                                #     data_bin = 0. * data
                                #     data_bin[data > self.eps] = 1.
                                #     fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                                #     trans_data.append(fake_data)
                                # else:
                                trans_data.append(gen_data)
                            else:
                                trans_data.append(data)

                        if self.ref_prior:
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(
                                self.device
                            )
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                        else:
                            c_bin = 0.0
                            prior_c = 0.0

                        self.optimizer.zero_grad()
                        recon_batch, p_x, r_x, x_low, qz, s, z, mu, log_var, log_qz = (
                            self.model(
                                trans_data, self.temp, prior_c, mask=pruning_mask
                            )
                        )
                        (
                            loss,
                            loss_rec,
                            loss_joint,
                            entropy,
                            dist_z,
                            d_qz,
                            KLD_cont,
                            min_var_0,
                            _,
                        ) = self.model.loss(
                            recon_batch, p_x, r_x, trans_data, mu, log_var, qz, z, c_bin
                        )

                        loss.backward()
                        self.optimizer.step()
                        ti[batch_indx] = time.time() - tt
                        train_loss_val += loss.data.item()
                        train_jointloss_val += loss_joint
                        train_dqz += d_qz
                        log_dqz += dist_z
                        entr += entropy
                        var_min += min_var_0.data.item()

                        for a in range(A):
                            train_loss_rec[a] += loss_rec[a].item() / D

                    train_loss[epoch] = train_loss_val / (batch_indx + 1)
                    train_loss_joint[epoch] = train_jointloss_val / (batch_indx + 1)
                    train_distance[epoch] = train_dqz / (batch_indx + 1)
                    train_entropy[epoch] = c_ent / (batch_indx + 1)
                    train_log_distance[epoch] = log_dqz / (batch_indx + 1)
                    train_minVar[epoch] = var_min / (batch_indx + 1)

                    for a in range(A):
                        train_recon[a, epoch] = train_loss_rec[a] / (batch_indx + 1)
                        for c in range(C):
                            train_loss_KL[a, c, epoch] = train_KLD_cont[a, c] / (
                                batch_indx + 1
                            )

                    print(
                        "====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {"
                        ":.4f}, Rec_arm_2: {:.4f}, Joint Loss: {:.4f}, Entropy: {:.4f}, Distance: {:.4f}, Elapsed Time:{:.2f}".format(
                            epoch,
                            train_loss[epoch],
                            train_recon[0, epoch],
                            train_recon[1, epoch],
                            train_loss_joint[epoch],
                            train_entropy[epoch],
                            train_distance[epoch],
                            time.time() - t0,
                        )
                    )

                    # validation
                    self.model.eval()
                    with th.no_grad():
                        val_loss_rec = 0.0
                        val_loss = 0.0
                        if test_loader.batch_size > 1:
                            for (
                                batch_indx,
                                (data_val, d_idx),
                            ) in enumerate(test_loader):
                                d_idx = d_idx.to(int)
                                data_val = data_val.to(self.device)

                                trans_val_data = []
                                for arm in range(self.n_arm):
                                    trans_val_data.append(data_val)

                                if self.ref_prior:
                                    c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(
                                        self.device
                                    )
                                    prior_c = torch.FloatTensor(c_p[d_idx, :]).to(
                                        self.device
                                    )
                                else:
                                    c_bin = 0.0
                                    prior_c = 0.0

                                (
                                    recon_batch,
                                    p_x,
                                    r_x,
                                    x_low,
                                    qc,
                                    s,
                                    c,
                                    mu,
                                    log_var,
                                    _,
                                ) = self.model(
                                    x=trans_val_data,
                                    temp=self.temp,
                                    prior_c=prior_c,
                                    eval=True,
                                    mask=pruning_mask,
                                )
                                loss, loss_rec, loss_joint, _, _, _, _, _, _ = (
                                    self.model.loss(
                                        recon_batch,
                                        p_x,
                                        r_x,
                                        trans_val_data,
                                        mu,
                                        log_var,
                                        qc,
                                        c,
                                        c_bin,
                                    )
                                )
                                val_loss += loss.data.item()
                                for arm in range(self.n_arm):
                                    val_loss_rec += (
                                        loss_rec[arm].data.item() / self.input_dim
                                    )
                        else:
                            batch_indx = 0
                            data_val, d_idx = test_loader.dataset.tensors
                            data_val = data_val.to(self.device)
                            d_idx = d_idx.to(int)
                            trans_val_data = []
                            for arm in range(self.n_arm):
                                trans_val_data.append(data_val)

                            if self.ref_prior:
                                c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(
                                    self.device
                                )
                                prior_c = torch.FloatTensor(c_p[d_idx, :]).to(
                                    self.device
                                )
                            else:
                                c_bin = 0.0
                                prior_c = 0.0

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = (
                                self.model(
                                    x=trans_val_data,
                                    temp=self.temp,
                                    prior_c=prior_c,
                                    eval=True,
                                    mask=pruning_mask,
                                )
                            )
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = (
                                self.model.loss(
                                    recon_batch,
                                    p_x,
                                    r_x,
                                    trans_val_data,
                                    mu,
                                    log_var,
                                    qc,
                                    c,
                                    c_bin,
                                )
                            )
                            val_loss = loss.data.item()
                            for a in range(A):
                                val_loss_rec += loss_rec[a].data.item() / D

                    validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / A
                    total_val_loss[epoch] = val_loss / (batch_indx + 1)
                    print(
                        "====> Validation Total Loss: {:.4}, Rec. Loss: {:.4f}".format(
                            total_val_loss[epoch], validation_rec_loss[epoch]
                        )
                    )

                for a in range(A):
                    prune.remove(self.model.fcc[a], "weight")
                    prune.remove(self.model.fcc[a], "bias")
                    prune.remove(self.model.fc_mu[a], "weight")
                    prune.remove(self.model.fc_sigma[a], "weight")
                    prune.remove(self.model.fc6[a], "weight")

                trained_model = (
                    self.folder
                    + "/model/cpl_mixVAE_model_after_pruning_"
                    + str(pr + 1)
                    + "_"
                    + self.current_time
                    + ".pth"
                )
                th.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    trained_model,
                )
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch_p), train_loss, label="Training")
                ax.plot(range(n_epoch_p), total_val_loss, label="Validation")
                ax.set_xlabel("# epoch", fontsize=16)
                ax.set_ylabel("loss value", fontsize=16)
                ax.set_title(
                    "Learning curve of the cpl-mixVAE for K="
                    + str(self.n_categories)
                    + " and S="
                    + str(self.state_dim)
                )
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.legend()
                ax.figure.savefig(
                    self.folder
                    + "../model/learning_curve_after_pruning_"
                    + str(pr + 1)
                    + "_K_"
                    + str(self.n_categories)
                    + "_"
                    + self.current_time
                    + ".png"
                )
                plt.close("all")
                pr += 1

        print("Training is done!")

        # return trained_model

    def eval_model(self, dl: DataLoader, c_p=0, c_onehot=0):
        """
        run the training of the cpl-mixVAE with the pre-defined parameters/settings
        pcikle used for saving the file

        input args
            data_loader: input data loader
            c_p: the prior categorical variable, only if ref_prior is True.
            c_onehot: the one-hot representation of the prior categorical variable, only if ref_prior is True.

        return
            d_dict: the output dictionary.
        """

        A = self.n_arm
        C = self.n_categories
        N = len(dl.dataset)
        D = self.input_dim
        D_low = self.lowD_dim
        S = self.state_dim
        B = unwrap(dl.batch_size)

        # Set the model to evaluation mode
        self.model.eval()

        # Extract bias and pruning mask
        bias = to_np(self.model.fcc[0].bias)
        pruning_mask = np.where(bias != 0.0)[0]
        prune_indx = np.where(bias == 0.0)[0]

        # Initialize arrays for storing evaluation results
        x_recs = np.zeros((A, N, D))
        s_means = np.zeros((A, N, S))
        s_logvars = np.zeros((A, N, S))
        cs = np.zeros((A, N, C))
        c_smps = np.zeros((A, N, C))
        x_lows = np.zeros((A, N, D_low))
        state_cat = np.zeros([A, N])
        prob_cat = np.zeros([A, N])
        predicted_label = np.zeros((A + self.ref_prior, N))

        data_indx = np.zeros(N)
        losses = []
        c_dists = []
        c_l2_dists = []
        loss_recs = [[] for _ in range(A)]
        lls = [[] for _ in range(A)]

        # Perform evaluation
        self.model.eval()
        with th.no_grad():
            for i, (x, data_idx) in enumerate(dl):
                n_fst = i * B
                n_lst = min((i + 1) * B, N)
                x = x.to(self.device)
                data_idx = data_idx.to(int)

                if self.ref_prior:
                    c_bin = th.tensor(
                        c_onehot[data_idx, :], dtype=th.float, device=self.device
                    )
                    c_prior = th.tensor(
                        c_p[data_idx, :], dtype=th.float, device=self.device
                    )
                else:
                    c_bin = 0.0
                    c_prior = 0.0

                xs = [x for _ in range(A)]

                (
                    _x_recs,
                    p_x,
                    r_x,
                    _x_lows,
                    _cs,
                    _s_smps,
                    _c_smps,
                    _s_means,
                    _s_logvars,
                    _,
                ) = self.model(
                    xs, self.temp, prior_c=c_prior, eval=True, mask=pruning_mask
                )
                _loss, _loss_recs, _, _, _c_dists, _c_l2_dists, _, _, _lls = (
                    self.model.loss(
                        _x_recs, p_x, r_x, xs, _s_means, _s_logvars, _cs, _c_smps, c_bin
                    )
                )
                losses.append(_loss.item() if isinstance(_loss, th.Tensor) else _loss)
                c_dists.append(
                    _c_dists.item() if isinstance(_c_dists, th.Tensor) else _c_dists
                )
                c_l2_dists.append(
                    _c_l2_dists.item()
                    if isinstance(_c_l2_dists, th.Tensor)
                    else _c_l2_dists
                )

                if self.ref_prior:
                    predicted_label[0, n_fst:n_lst] = (
                        np.argmax(c_p[data_idx, :], axis=1) + 1
                    )

                for a, (loss_rec, ll) in enumerate(zip(_loss_recs, _lls)):
                    loss_recs[a].append(loss_rec.item())
                    lls[a].append(ll.item())

                for a, (s_mean, s_logvar, c, c_smp, x_low, x_rec) in enumerate(
                    map(
                        lambda ys: map(to_np, ys),
                        zip(_s_means, _s_logvars, _cs, _c_smps, _x_lows, _x_recs),
                    )
                ):
                    s_means[a, n_fst:n_lst, :] = s_mean
                    s_logvars[a, n_fst:n_lst, :] = s_logvar
                    cs[a, n_fst:n_lst, :] = c
                    c_smps[a, n_fst:n_lst, :] = c_smp
                    x_lows[a, n_fst:n_lst, :] = x_low
                    x_recs[a, n_fst:n_lst, :] = x_rec
                    data_indx[n_fst:n_lst] = data_idx.numpy().astype(int)
                    for n, c_n in enumerate(c):
                        state_cat[a, n_fst + n] = np.argmax(c_n) + 1
                        prob_cat[a, n_fst + n] = np.max(c_n)
                    if self.ref_prior:
                        predicted_label[a + 1, n_fst:n_lst] = np.argmax(c, axis=-1) + 1
                    else:
                        predicted_label[a, n_fst:n_lst] = np.argmax(c, axis=-1) + 1

        return {
            "state_mu": s_means,
            "state_var": s_logvars,
            "state_cat": state_cat,
            "prob_cat": prob_cat,
            "total_loss_rec": np.array(
                [np.mean(np.array(loss_recs[a])) for a in range(A)]
            ),
            "total_likelihood": np.array([np.mean(np.array(lls[a])) for a in range(A)]),
            "total_dist_z": np.mean(np.array(c_dists)),
            "total_dist_qz": np.mean(np.array(c_l2_dists)),
            "mean_test_rec": np.zeros(A),
            "predicted_label": predicted_label,
            "data_indx": data_indx,
            "z_prob": cs,
            "z_sample": c_smps,
            "x_low": x_lows,
            "recon_c": x_recs,
            "prune_indx": prune_indx,
        }

    def save_file(self, fname, **kwargs):
        """
        Save data as a .p file using pickle.

        input args
            fname: the path of the pre-trained network.
            kwarg: keyword arguments for input variables e.g., x=[], y=[], etc.
        """

        f = open(fname + ".p", "wb")
        data = {}
        for k, v in kwargs.items():
            data[k] = v
        pickle.dump(data, f, protocol=4)
        f.close()

    def load_file(self, fname):
        """
        load data .p file using pickle. Make sure to use the same version of
        pcikle used for saving the file

        input args
            fname: the path of the pre-trained network.

        return
            data: a dictionary including the save dataset
        """

        data = pickle.load(open(fname + ".p", "rb"))
        return data


# 50 epochs
