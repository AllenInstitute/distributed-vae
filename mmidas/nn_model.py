from dataclasses import dataclass
from typing import Optional, List, Iterable, Sequence, assert_never

import numpy as np
import torch
import torch as th
from torch import nn
from torch.nn import ModuleList as mdl
from torch.autograd import Variable
from torch.nn import functional as F


# 2 arm: 0, 5, 6
# 3 arm: 0, 1, 3, 4 (need to find correct run for 0, 1)
# 5 arm: 1, 3, 4, 5

@dataclass
class VAEConfig:
    n_categories: int = 92
    state_dim: int = 2
    input_dim: int = 5032
    fc_dim: int = 100
    lowD_dim: int = 10
    x_drop: float = 0.5
    s_drop: float = 0.2
    lr: float = 0.001
    lam: float = 1
    lam_pc: float = 1
    n_arm: int = 2
    temp: float = 1.0
    tau: float = 0.005
    beta: float = 1.0
    hard: bool = False
    variational: bool = True
    ref_prior: bool = False
    trained_model: Optional[str] = None
    n_pr: int = 0
    momentum: float = 0.01
    mode: str = "MSE"


def binarize(x, eps):
    return th.where(x > eps, 1.0, 0.0)


def kl(mean: th.Tensor, logvar: th.Tensor) -> th.Tensor:
    return (-0.5 * th.mean(1 + logvar - mean.pow(2) - logvar.exp(), dim=0)).sum()


def arm_combs(A: int) -> int:
    if A > 1:
        assert (A * (A - 1)) % 2 == 0
    return max(A * (A - 1) / 2, 1)


def l2_dist(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    return th.norm(a - b, p=2, dim=-1).pow(2)


def simplex_dist(
    a: tuple[th.Tensor, th.Tensor], b: tuple[th.Tensor, th.Tensor]
) -> th.Tensor:
    loga, inv_vara = a
    logb, inv_varb = b
    return l2_dist(loga * inv_vara, logb * inv_varb)


def neg_entropy(p: th.Tensor, logp: th.Tensor) -> th.Tensor:
    return th.sum(p * logp, dim=-1)


def neg_joint_entropy(
    a: tuple[th.Tensor, th.Tensor], b: tuple[th.Tensor, th.Tensor]
) -> th.Tensor:
    return neg_entropy(*a).mean() + neg_entropy(*b).mean()


def inv_var(p: th.Tensor, eps: float) -> th.Tensor:
    return (1 / (p.var(0) + eps)).repeat(p.size(0), 1).sqrt()


def avg[T](x: Sequence[T]) -> T:
    return sum(x) / len(x)


class mixVAE_model(nn.Module):
    """
    Class for the neural network module for mixture of continuous and
    discrete random variables. The module contains an VAE using
    Gumbel-softmax distribution for the categorical and reparameterization
    for continuous latent variables.
    The default setting of this network is for smart-seq datasets. If you
    want to use another dataset, you may need to modify the network's
    parameters.

    Methods
        encoder: encoder network.
        intermed: the intermediate layer for combining categorical and continuous RV.
        decoder: decoder network.
        forward: module for forward path.
        state_changes: module for the continues variable analysis
        reparam_trick: module for reparameterization.
        sample_gumbel: samples by adding Gumbel noise.
        gumbel_softmax_sample: Gumbel-softmax sampling module
        gumbel_softmax: Gumbel-softmax distribution module
        loss: loss function module
    """

    def __init__(
        self,
        input_dim,
        fc_dim,
        n_categories,
        state_dim,
        lowD_dim,
        x_drop,
        s_drop,
        n_arm,
        lam,
        lam_pc,
        tau,
        beta,
        hard,
        variational,
        device,
        eps,
        momentum,
        ref_prior,
        loss_mode,
    ):
        """
        Class instantiation.

        input args
            input_dim: input dimension (size of the input layer).
            fc_dim: dimension of the hidden layer.
            n_categories: number of categories of the latent variables.
            state_dim: dimension of the continuous (state) latent variable.
            lowD_dim: dimension of the latent representation.
            x_drop: dropout probability at the first (input) layer.
            s_drop: dropout probability of the state variable.
            n_arm: int value that indicates number of arms.
            lam: coupling factor in the cpl-mixVAE model.
            lam_pc: coupling factor for the prior categorical variable.
            tau: temperature of the softmax layers, usually equals to 1/n_categories (0 < tau <= 1).
            beta: regularizer for the KL divergence term.
            hard: a boolean variable, True uses one-hot method that is used in Gumbel-softmax, and False uses the Gumbel-softmax function.
            variational: a boolean variable for variational mode, False mode does not use sampling.
            device: computing device, either 'cpu' or 'cuda'.
            eps: a small constant value to fix computation overflow.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
            ref_prior: a boolean variable, True uses the reference prior for the categorical variable.
            loss_mode: string, define the reconstruction loss function, either MSE or ZINB.
        """
        super().__init__()
        self.input_dim = input_dim
        self.fc_dim = fc_dim
        self.lowD_dim = lowD_dim
        self.state_dim = state_dim
        self.n_categories = n_categories
        self.x_dp = nn.Dropout(x_drop)
        self.s_dp = nn.Dropout(s_drop)
        self.hard = hard
        self.n_arm = n_arm
        self.lam = lam
        self.lam_pc = lam_pc
        self.tau = tau
        self.beta = beta
        self.varitional = variational
        self.eps = eps
        self.ref_prior = ref_prior
        self.momentum = momentum
        self.device = device
        self.loss_mode = loss_mode

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        self.fc1 = mdl([nn.Linear(input_dim, fc_dim) for i in range(n_arm)])
        self.fc2 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc3 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc4 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc5 = mdl([nn.Linear(fc_dim, lowD_dim) for i in range(n_arm)])
        self.fcc = mdl([nn.Linear(lowD_dim, n_categories) for i in range(n_arm)])
        self.fc_mu = mdl(
            [nn.Linear(lowD_dim + n_categories, state_dim) for i in range(n_arm)]
        )
        self.fc_sigma = mdl(
            [nn.Linear(lowD_dim + n_categories, state_dim) for i in range(n_arm)]
        )
        self.fc6 = mdl(
            [nn.Linear(state_dim + n_categories, lowD_dim) for i in range(n_arm)]
        )
        self.fc7 = mdl([nn.Linear(lowD_dim, fc_dim) for i in range(n_arm)])
        self.fc8 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc9 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc10 = mdl([nn.Linear(fc_dim, fc_dim) for i in range(n_arm)])
        self.fc11 = mdl([nn.Linear(fc_dim, input_dim) for i in range(n_arm)])
        if loss_mode == "ZINB":
            self.fc11_p = mdl([nn.Linear(fc_dim, input_dim) for i in range(n_arm)])
            self.fc11_r = mdl([nn.Linear(fc_dim, input_dim) for i in range(n_arm)])

        self.batch_l1 = mdl(
            [
                nn.BatchNorm1d(
                    num_features=fc_dim, eps=eps, momentum=momentum, affine=False
                )
                for i in range(n_arm)
            ]
        )
        self.batch_l2 = mdl(
            [
                nn.BatchNorm1d(
                    num_features=fc_dim, eps=eps, momentum=momentum, affine=False
                )
                for i in range(n_arm)
            ]
        )
        self.batch_l3 = mdl(
            [
                nn.BatchNorm1d(
                    num_features=fc_dim, eps=eps, momentum=momentum, affine=False
                )
                for i in range(n_arm)
            ]
        )
        self.batch_l4 = mdl(
            [
                nn.BatchNorm1d(
                    num_features=fc_dim, eps=eps, momentum=momentum, affine=False
                )
                for i in range(n_arm)
            ]
        )
        self.batch_l5 = mdl(
            [
                nn.BatchNorm1d(
                    num_features=lowD_dim, eps=eps, momentum=momentum, affine=False
                )
                for i in range(n_arm)
            ]
        )
        self.batch_s = mdl(
            [
                nn.BatchNorm1d(
                    num_features=state_dim, eps=eps, momentum=momentum, affine=False
                )
                for i in range(n_arm)
            ]
        )

        self.c_var_inv = [None] * 2
        self.stack_mean = [[] for a in range(2)]
        self.stack_var = [[] for a in range(2)]
        self.c_mean = [None] * 2
        self.c_var = [None] * 2

    def encoder(self, x, arm):
        x = self.batch_l1[arm](F.relu(self.fc1[arm](self.x_dp(x))))
        x = self.batch_l2[arm](F.relu(self.fc2[arm](x)))
        x = self.batch_l3[arm](F.relu(self.fc3[arm](x)))
        x = self.batch_l4[arm](F.relu(self.fc4[arm](x)))
        logits = self.batch_l5[arm](F.relu(self.fc5[arm](x)))
        return logits, F.softmax(self.fcc[arm](logits), dim=-1)

    def intermed(self, x, arm):
        if self.varitional:
            return self.fc_mu[arm](x), th.sigmoid(self.fc_sigma[arm](x))
        else:
            return self.fc_mu[arm](x)

    def _decode(self, c, s, arm):
        s = self.s_dp(s)
        z = th.cat((c, s), dim=1)
        x = F.relu(self.fc6[arm](z))
        x = F.relu(self.fc7[arm](x))
        x = F.relu(self.fc8[arm](x))
        x = F.relu(self.fc9[arm](x))
        return F.relu(self.fc10[arm](x))

    def decoder(self, c, s, arm):
        return F.relu(self.fc11[arm](self._decode(c, s, arm)))

    def decoder_zinb(self, c, s, arm):
        x = self._decode(c, s, arm)
        return (
            F.relu(self.fc11[arm](x)),
            th.sigmoid(self.fc11_p[arm](x)),
            th.sigmoid(self.fc11_r[arm](x)),
        )

    def forward(self, x, temp, prior_c=[], eval=False, mask=None):
        """
        input args
            x: a list including input batch tensors (batch size x number of features) for each arm.
            temp: temperature of Gumbel-softmax function.
            eval: a boolean variable, set True for during evaluation.
            mask: masked unrequired categorical variable at the time of pruning.

        return
            recon_x: a list including the reconstructed data for each arm.
            x_low: a list including a low dimensional representation of the input for each arm.
            qc: list of pdf of the categorical variable for all arms.
            s: list of sample of the sate variable for all arms.
            c: list of sample of the categorical variable for all arms.
            mu: list of mean of the state variable for all arms.
            log_var: list of log of variance of the state variable for all arms.
            log_qc: list of log-likelihood value of categorical variables in a batch for all arms.
        """
        assert not self.loss_mode == "ZINB", "ZINB not implemented"
        assert self.varitional, "Non-variational not implemented"

        C = self.n_categories

        xs = x
        c_prior = prior_c

        x_recs, x_lows = [], []
        cs = []
        s_smps, c_smps = [], []
        s_means, s_logvars = [], []
        c_probs = []
        for a, x in enumerate(xs):  # arm, data
            x_low, c_prob = self.encoder(x, a)

            if mask is not None:
                c_tmp = F.softmax(c_prob[:, mask] / self.tau, dim=-1)
                c = th.zeros((c_prob.size(0), c_prob.size(1)), device=self.device)
                c[:, mask] = c_tmp
            else:
                c = F.softmax(c_prob / self.tau, dim=-1)

            logits = c.view(c_prob.size(0), 1, C)
            if eval:
                c_smp = self.gumbel_softmax(
                    logits, 1, C, temp, hard=True, gumble_noise=False
                )
            else:
                c_smp = self.gumbel_softmax(logits, 1, C, temp, hard=self.hard)

            y = th.cat((x_low, c_prior if self.ref_prior else c_smp), dim=1)
            if self.varitional:
                s_mean, s_var = self.intermed(y, a)
                s_logvar = (s_var + self.eps).log()
                s_smp = self.reparam_trick(s_mean, s_logvar)
            else:
                s_mean = self.intermed(y, a)
                s_logvar = 0.0 * s_mean
                s_smp = self.intermed(y, a)

            x_rec = self.decoder(c_smp, s_smp, a)

            x_recs.append(x_rec)
            x_lows.append(x_low)
            cs.append(c)
            s_smps.append(s_smp)
            c_smps.append(c_smp)
            s_means.append(s_mean)
            s_logvars.append(s_logvar)
            c_probs.append(c_prob)

        return x_recs, [], [], x_lows, cs, s_smps, c_smps, s_means, s_logvars, c_probs

    def state_changes(self, x, d_s, temp, n_samp=100):
        """
        Continuous traversal study.

        input args
            x: input batch tensor (batch size x dimension of features).
            d_s: selected dimension of the state variable.
            temp: temperature of Gumbel-softmax function.
            n_samp: number of samples for the continues traversal study.

        return
            recon_x: 3D tensor including reconstructed data for all arms.
            state_smp_sorted: 2D tensor including sorted continues variable samples for all arms.
        """
        state_var = np.linspace(-0.01, 0.01, n_samp)
        recon_x = torch.zeros((self.n_arm, len(state_var), x.size(-1)))
        var_state = torch.zeros((len(state_var)))
        state_smp_sorted = torch.zeros((self.n_arm, len(state_var)))

        for arm in range(self.n_arm):
            x_low, q = self.encoder(x, arm)
            q = F.softmax(q / self.tau, dim=-1)
            q_c = q.view(q.size(0), 1, self.n_categories)
            c = self.gumbel_softmax(
                q_c, 1, self.n_categories, temp, hard=True, gumble_noise=False
            )
            y = torch.cat((x_low, c), dim=1)
            if self.varitional:
                mu, log_var = self.intermed(y, arm)
            else:
                mu = self.intermed(y, arm)
                log_var = 0.0

            for i in range(len(state_var)):
                s = mu.clone()
                s[:, d_s] = self.reparam_trick(mu[:, d_s], log_var[:, d_s].log())
                recon_x[arm, i, :] = self.decoder(c, s, arm)

            state_smp_sorted[arm, :], sort_idx = var_state.sort()
            recon_x[arm, :, :] = recon_x[arm, sort_idx, :]

        return recon_x, state_smp_sorted

    def reparam_trick(self, mu, log_sigma):
        """
        Generate samples from a normal distribution for reparametrization trick.

        input args
            mu: mean of the Gaussian distribution for
                q(s|z,x) = N(mu, sigma^2*I).
            log_sigma: log of variance of the Gaussian distribution for
                       q(s|z,x) = N(mu, sigma^2*I).

        return
            a sample from Gaussian distribution N(mu, sigma^2*I).
        """
        std = log_sigma.exp().sqrt()
        eps = torch.rand_like(std).to(self.device)
        return eps.mul(std).add(mu)

    def sample_gumbel(self, shape):
        """
        Generates samples from Gumbel distribution.

        input args
            size: number of cells in a batch (int).

        return
            -(log(-log(U))) (tensor)
        """
        U = th.rand(shape, device=self.device)
        return -Variable(th.log(-th.log(U + self.eps) + self.eps))

    def gumbel_softmax_sample(self, phi, temperature):
        """
        Generates samples via Gumbel-softmax distribution.

        input args
            phi: probabilities of categories.
            temperature: a hyperparameter that define the shape of the distribution across categtories.

        return
            Samples from a categorical distribution.
        """
        logits = (phi + self.eps).log() + self.sample_gumbel(phi.size())
        return F.softmax(logits / temperature, dim=-1)

    def gumbel_softmax(
        self,
        phi,
        latent_dim,
        categorical_dim,
        temperature,
        hard=False,
        gumble_noise=True,
    ):
        """
        Implements Straight-Through (ST) Gumbel-softmax and regular Gumbel-softmax.

        input args
            phi: probabilities of categories.
            latent_dim: latent variable dimension.
            categorical_dim: number of categories of the latent variables.
            temperature: a hyperparameter that define the shape of the distribution across categories.
            hard: a boolean variable, True uses one-hot method that is used in ST Gumbel-softmax, and False uses the Gumbel-softmax function.

        return
            Samples from a categorical distribution, a tensor with latent_dim x categorical_dim.
        """
        if gumble_noise:
            y = self.gumbel_softmax_sample(phi, temperature)
        else:
            y = phi

        if not hard:
            return y.view(-1, latent_dim * categorical_dim)
        else:
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            y_hard = (y_hard - y).detach() + y
            return y_hard.view(-1, latent_dim * categorical_dim)

    def loss(self, recon_x, p_x, r_x, x, mu, log_sigma, qc, c, prior_c=[]):
        """
         loss function of the cpl-mixVAE network including.

        input args
             recon_x: a list including the reconstructed data for each arm.
             x: a list includes original input data.
             mu: list of mean of the Gaussian distribution for the sate variable.
             log_sigma: log of variance of the Gaussian distribution for the sate variable.
             qc: probability of categories for all arms.
             c: samples fom all distrubtions for all arms.
             prior_c: prior probability of the categories for all arms, if ref_prior is True.

         return
             total_loss: total loss value.
             l_rec: reconstruction loss for each arm.
             loss_joint: coupling loss.
             neg_joint_entropy: negative joint entropy of the categorical variable.
             qc_distance: distance between a pair of categorical distributions, i.e. qc_a & qc_b.
             c_distance: Euclidean distance between a pair of categorical variables, i.e. c_a & c_b.
             KLD: list of KL divergences for the state variables across all arms.
             var_a.min(): minimum variance of the last arm.
             loglikelihood: list of log-likelihood values for all arms

        """
        A = self.n_arm
        C = self.n_categories
        B = x[0].size(0)

        xs = x
        x_recs = recon_x
        s_means = mu
        s_logvars = log_sigma
        _c = qc
        c_smps = c
        c_prior = prior_c

        lls = []  # log-likelihood
        loss_recs, loss_inds, kl_ss = [], [], []
        c_ents, c_l2_dists, c_dists = [], [], []

        # TODO: this loop is really easy to parallelize/refactor with attention
        # q, k, v = ((c, logc, inv_var_c), (c, logc, inv_var_c), (c, logc, inv_var_c)). maybe add kernels too?
        for a, (x, x_rec, s_mean, s_logvar, c_a, c_smp_a) in enumerate(
            zip(xs, x_recs, s_means, s_logvars, _c, c_smps)
        ):  # a ∈ 0..A-1
            ll = F.mse_loss(x_rec, x, reduction="mean") + B * np.log(2 * np.pi)
            if self.loss_mode == "MSE":
                loss_rec = (0.5 * F.mse_loss(x_rec, x, reduction="sum") / B) + (
                    0.5 * F.binary_cross_entropy(binarize(x_rec, 0.1), binarize(x, 0.1))
                )
            elif self.loss_mode == "ZINB":
                assert False, "ZINB not implemented"
                loss_rec = zinb_loss(x_rec, p_x[a], r_x[a], x)
            kl_s = kl(s_mean, s_logvar) if self.varitional else [0.0]
            loss_ind = loss_rec + self.beta * kl_s

            lls.append(ll)
            loss_recs.append(loss_rec)
            kl_ss.append(kl_s)
            loss_inds.append(loss_ind)

            logc_a = th.log(c_a + self.eps)
            inv_var_c_a = inv_var(c_a, self.eps)

            for c_b, c_smp_b in zip(_c[a + 1 :], c_smps[a + 1 :]):  # b ∈ a+1..A-1
                logc_b = th.log(c_b + self.eps)
                inv_var_c_b = inv_var(c_b, self.eps)

                c_ents.append(neg_joint_entropy((c_a, logc_a), (c_b, logc_b)))
                c_l2_dists.append(l2_dist(c_smp_a, c_smp_b).mean())
                c_dists.append(
                    simplex_dist((logc_a, inv_var_c_a), (logc_b, inv_var_c_b)).mean()
                )

            if self.ref_prior:
                c_bin = self.gumbel_softmax(c_a, 1, C, 1, hard=True, gumble_noise=False)

                c_ents.append(neg_entropy(c_a, logc_a).mean())
                c_l2_dists.append(l2_dist(c_smp_a, c_prior).mean())
                c_dists.append(self.lam_pc * F.binary_cross_entropy(c_bin, c_prior))

        assert not self.ref_prior
        sum_c_dists = sum(c_dists)
        sum_c_ents = sum(c_ents)
        loss_joints = (
            self.lam * sum_c_dists
            + sum_c_ents
            + arm_combs(A)
            * ((C / 2) * (np.log(2 * np.pi)) - 0.5 * np.log(2 * self.lam))
        )
        losses = max((A - 1), 1) * sum(loss_inds) + loss_joints
        return (
            losses,
            th.tensor(loss_recs, device=self.device),
            loss_joints,
            sum_c_ents / len(c_ents),
            sum_c_dists / len(c_dists),
            avg(c_l2_dists),
            kl_ss,
            [],
            lls,
        )
    
    def loss_vectorized(self, x_recs, p_x, r_x, xs, s_means, s_logvars, c, c_smps):
        assert len(x_recs) == len(xs) == len(s_means) == len(s_logvars) == len(c_smps) == self.n_arm
        assert not self.ref_prior

        A = self.n_arm
        K = self.n_categories
        (B, _) = xs[0].shape

        lls = []
        loss_recs = []
        loss_inds = []
        kl_ss = []
        c_ents = []
        c_l2_dists = []
        c_dists = []
        for (a, (x, x_rec, s_mean, s_logvar, c_a, c_smp_a)) in enumerate(zip(xs, x_recs, s_means, s_logvars, c, c_smps)):
            ll = F.mse_loss(x_rec, x, reduction="mean") + B * np.log(2 * np.pi)

            if self.loss_mode == "MSE":
                loss_rec = (0.5 * F.mse_loss(x_rec, x, reduction="sum") / B) + (0.5 * F.binary_cross_entropy(binarize(x_rec, 0.1), binarize(x, 0.1)))
            elif self.loss_mode == "ZINB":
                print("warning: ZINB is unstable")
                loss_rec = zinb_loss(x_rec, p_x[a], r_x[a], x)
            else:
                assert_never(self.loss_mode)

            if self.varitional:
                kl_s = kl(s_mean, s_logvar)
            else:
                kl_s = [0.0]

            loss_ind = loss_rec + self.beta * kl_s

            lls.append(ll)
            loss_recs.append(loss_rec)
            kl_ss.append(kl_s)
            loss_inds.append(loss_ind)

            logc_a = th.log(c_a + self.eps)
            inv_var_c_a = inv_var(c_a, self.eps)
            for (c_b, c_smp_b) in zip(c[a + 1:], c_smps[a + 1:]):
                logc_b = th.log(c_b + self.eps)
                inv_var_c_b = inv_var(c_b, self.eps)

                c_ents.append(neg_joint_entropy((c_a, logc_a), (c_b, logc_b)))
                c_l2_dists.append(l2_dist(c_smp_a, c_smp_b).mean())
                c_dists.append(simplex_dist((logc_a, inv_var_c_a), (logc_b, inv_var_c_b)).mean())

        sum_c_dists = sum(c_dists)
        sum_c_ents = sum(c_ents)
        loss_joints = self.lam * sum_c_dists + sum_c_ents + arm_combs(A) * ((K / 2) * (np.log(2 * np.pi)) - 0.5 * np.log(2 * self.lam))




def zinb_loss(rec_x, x_p, x_r, X, eps=1e-6):
    """
     loss function using zero inflated negative binomial distribution for
     log(x|s,z) for genes expression data.

    input args
         rec_x: log of mean value of the negative binomial distribution.
         x_p: log of the probability of dropout events.
         x_r: log of the probability of zero inflation.
         X: input data.
         eps: a small constant value to fix computation overflow.

     return
         l_zinb: log of loss value
    """

    X_dim = X.size(-1)
    k = X.exp() - 1.0  # logp(count) -->  (count)

    # extracting r,p, and z from the concatenated vactor.
    # eps added for stability.
    r = rec_x + eps  # zinb_params[:, :X_dim] + eps
    p = (1 - eps) * (x_p + eps)  # (1 - eps)*(zinb_params[:, X_dim:2*X_dim] + eps)
    z = (1 - eps) * (x_r + eps)  # (1 - eps)*(zinb_params[:, 2*X_dim:] + eps)

    mask_nonzeros = ([X > 0])[0].to(torch.float32)
    loss_zero_counts = (mask_nonzeros - 1) * (z + (1 - z) * (1 - p).pow(r)).log()
    # log of zinb for non-negative terms, excluding x! term
    loss_nonzero_counts = mask_nonzeros * (
        -(k + r).lgamma() + r.lgamma() - k * p.log() - r * (1 - p).log() - (1 - z).log()
    )

    l_zinb = (loss_zero_counts + loss_nonzero_counts).mean()

    return l_zinb
