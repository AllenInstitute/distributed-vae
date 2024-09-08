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
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm, trange
import wandb

from .augmentation.udagan import *
from .nn_model import mixVAE_model, VAEConfig
from .utils.data_tools import split_data_Kfold
from .utils.dataloader import get_sampler, is_dist_sampler

Device = Literal['cpu', 'cuda', 'mps'] | int

# TODO

@dataclass
class Master:
    rank: Literal[0, 'mps', 'cpu', 'cuda']

@dataclass
class Worker:
    rank: int

Rank = Master | Worker

def mk_rank(i: int | Literal['mps', 'cpu', 'cuda']) -> Rank:
    match i:
        case 0 | 'mps' | 'cpu' | 'cuda':
            return Master(i)
        case int(j):
            return Worker(j)
        case _:
            assert_never(i)

def is_master(rank):
    return rank == 0 or rank == 'mps' or rank == 'cpu' or isinstance(rank, Master)

def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    return reduce(compose2, fs)

def mk_pbar(seq: Sequence, r: Rank, *fs) -> Sequence:
    return (lambda x: tqdm(x, total=len(x)) if is_master(r) else x)(compose(*fs)(seq)) # type: ignore

    # _seq: Sequence = compose(*fs)(seq)
    # match r:
    #     case Master(_):
    #         return tqdm(_seq, total=len(seq)) # type: ignore
    #     case Worker(_):
    #         return _seq
    #     case _:
    #         assert_never(r)

    # loader = 
    # if is_master(rank):
    #     return tqdm(loader, total=len(loader), unit_scale=True)
    # else:
    #     return loader
    
def is_parallel(world_size):
    return world_size > 1
  
def print_train_loss(epoch, train_loss, train_recon0, train_recon1, train_loss_joint, train_entropy, train_distance, time, rank):
    if is_master(rank):
        print('====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {:.4f}, Rec_arm_2: {:.4f}, Joint Loss: {:.4f}, Entropy: {:.4f}, Distance: {:.4f}, Elapsed Time:{:.2f}'.format(
            epoch, train_loss, train_recon0, train_recon1, train_loss_joint, train_entropy, train_distance, time))
        
def print_val_loss(val_loss, val_loss_rec, rank):
    if is_master(rank):
        print('====> Validation Total Loss: {:.4f}, Rec. Loss: {:.4f}'.format(val_loss, val_loss_rec))
  
def avg_recon_loss(l: np.ndarray) -> float:
    return np.mean(l, axis=0)

def unwrap[T](x: Optional[T]) -> T:
    if x is None:
        raise ValueError("error: expected non-None value")
    return x

def len_cpus(deterministic=False) -> int:
    if deterministic:
        return 1
    elif hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    else:
        return unwrap(os.cpu_count())
    
def get_device(device: Optional[Device] = None) -> th.device:
    match device:
        case 'cpu' | 'mps' as d:
            print(f'using {d}')
            return th.device(d)
        case 'cuda' as d:
            print(f'using {d}: {cuda.get_device_name(d)}')
            return th.device(d)
        case int(d):
            cuda.set_device(d)
            return get_device('cuda')
        case None:
            print('device not found')
            return get_device('cpu')
        case _:
            assert_never(device)

def mk_augmenter(pretrained: str, load_weights: bool) -> tuple[Mapping[Any, Any], Mapping[Any, Any], nn.Module]:
    aug_model = th.load(pretrained, map_location='cpu')
    aug_param = aug_model['parameters']
    if load_weights:
        print('loading augmenter weights')
        netA = Augmenter_smartseq(noise_dim=aug_param['num_n'],
                                  latent_dim=aug_param['num_z'],
                                  input_dim=aug_param['n_features'])
        netA.load_state_dict(aug_model['netA'])
        return aug_model, aug_param, netA
    else:
        print('warning: not loading augmenter weights')
        netA = Augmenter(noise_dim=aug_param['num_n'],
                         latent_dim=aug_param['num_z'],
                         input_dim=aug_param['n_features'])
        return aug_model, aug_param, netA

def asnp(x):
    return x.cpu().detach().numpy()

class cpl_mixVAE:

    def __init__(self, saving_folder='', aug_file='', device: Optional[Device] = None, eps=1e-8, 
                 save_flag=True, load_weights=True):
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

    def get_dataloader(self, dataset, label, batch_size=128, n_aug_smp=0, k_fold=10, fold=0, rank=-1, world_size=-1, use_dist_sampler=False, deterministic=False):
        self.batch_size = batch_size


        train_inds, test_inds = split_data_Kfold(label, k_fold)
        train_ind = train_inds[fold].astype(int)
        test_ind = test_inds[fold].astype(int)

        train_set_torch = torch.FloatTensor(dataset[train_ind, :])
        train_ind_torch = torch.FloatTensor(train_ind)
        if n_aug_smp > 0:
            train_set = train_set_torch.clone()
            train_set_ind = train_ind_torch.clone()
            for n_a in range(n_aug_smp):
                if self.aug_file:
                    noise = torch.randn(train_set_torch.shape[0], self.aug_param['num_n'])
                    if self.gpu:
                        _, gen_data = self.netA(train_set_torch.cuda(self.device), noise.cuda(self.device), self.device)
                    else:
                        _, gen_data = self.netA(train_set_torch, noise, self.device)

                    train_set = torch.cat((train_set, gen_data.cpu().detach()), 0)

                else:
                    train_set = torch.cat((train_set, train_set_torch), 0)
                train_set_ind = torch.cat((train_set_ind, train_ind_torch), 0)

            train_data = TensorDataset(train_set, train_set_ind)
        else:
            train_data = TensorDataset(train_set_torch, train_ind_torch)

        if world_size > 1 and use_dist_sampler:
            train_sampler = DistributedSampler(train_data, rank=rank, num_replicas=world_size, shuffle=True)
            train_loader = DataLoader(train_data, batch_size=batch_size, 
                                      drop_last=True, pin_memory=True, 
                                      persistent_workers=True, num_workers=len_cpus(deterministic),
                                      sampler=train_sampler)
        else:
            train_loader = DataLoader(train_data, batch_size=batch_size, 
                                      drop_last=True, pin_memory=True, persistent_workers=True,
                                      num_workers=len_cpus(deterministic))

        val_set_torch = torch.FloatTensor(dataset[test_ind, :])
        val_ind_torch = torch.FloatTensor(test_ind)
        validation_data = TensorDataset(val_set_torch, val_ind_torch)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)

        test_set_torch = torch.FloatTensor(dataset[test_ind, :])
        test_ind_torch = torch.FloatTensor(test_ind)
        test_data = TensorDataset(test_set_torch, test_ind_torch)

        if world_size > 1 and use_dist_sampler:
            print('using distributed sampler...')
            test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=1, drop_last=False, pin_memory=True, persistent_workers=True, num_workers=len_cpus(deterministic),
                                    sampler=test_sampler)
        else:
            test_loader = DataLoader(test_data, batch_size=1, drop_last=True,
                                     pin_memory=True, persistent_workers=True,
                                     num_workers=len_cpus(deterministic))

        data_set_troch = torch.FloatTensor(dataset)
        all_ind_torch = torch.FloatTensor(range(dataset.shape[0]))
        all_data = TensorDataset(data_set_troch, all_ind_torch)
        alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        return alldata_loader, train_loader, validation_loader, test_loader

    def init_model(self, n_categories, state_dim, input_dim, fc_dim=100, lowD_dim=10, x_drop=0.5, s_drop=0.2, lr=.001,
                   lam=1, lam_pc=1, n_arm=2, temp=1., tau=0.005, beta=1., hard=False, variational=True, ref_prior=False,
                   trained_model='', n_pr=0, momentum=.01, mode='MSE'):
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
        self.model = mixVAE_model(input_dim=self.input_dim, fc_dim=fc_dim, n_categories=self.n_categories, state_dim=self.state_dim,
                                lowD_dim=lowD_dim, x_drop=x_drop, s_drop=s_drop, n_arm=self.n_arm, lam=lam, lam_pc=lam_pc,
                                tau=tau, beta=beta, hard=hard, variational=variational, device=self.device, eps=self.eps,
                                ref_prior=ref_prior, momentum=momentum, loss_mode=mode)
        
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if len(trained_model) > 0:
            print('Load the pre-trained model')
            # if you wish to load another model for evaluation
            loaded_file = torch.load(trained_model, map_location='cpu')
            self.model.load_state_dict(loaded_file['model_state_dict'])
            self.optimizer.load_state_dict(loaded_file['optimizer_state_dict'])
            self.init = False
            self.n_pr = n_pr
        else:
            self.init = True
            self.n_pr = 0

    def append(self, c: VAEConfig):
        model = mixVAE_model(input_dim=c.input_dim, fc_dim=c.fc_dim, n_categories=c.n_categories, state_dim=c.state_dim,
                                lowD_dim=c.lowD_dim, x_drop=c.x_drop, s_drop=c.s_drop, n_arm=c.n_arm, lam=c.lam, lam_pc=c.lam_pc,
                                tau=c.tau, beta=c.beta, hard=c.hard, variational=c.variational, device=self.device, eps=self.eps,
                                ref_prior=c.ref_prior, momentum=c.momentum, loss_mode=c.mode).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        if c.trained_model:
            loaded_file = th.load(c.trained_model, map_location='cpu')
            model.load_state_dict(loaded_file['model_state_dict'])
            optimizer.load_state_dict(loaded_file['optimizer_state_dict'])
        self.models.append({'model': model, 'opt': optimizer})
        
    def load_model(self, trained_model):
        loaded_file = torch.load(trained_model, map_location='cpu')
        self.model.load_state_dict(loaded_file['model_state_dict'])

        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

    def train(self, train_loader, test_loader, n_epoch, n_epoch_p, c_p=0, c_onehot=0, min_con=.5, max_prun_it=0, rank=None, run=None, ws=1):
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
        # define current_time
        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # initialized saving arrays

        train_loss = np.zeros(n_epoch)
        validation_loss = np.zeros(n_epoch)
        train_loss_joint = np.zeros(n_epoch)
        train_entropy = np.zeros(n_epoch)
        train_distance = np.zeros(n_epoch)
        train_minVar = np.zeros(n_epoch)
        train_log_distance = np.zeros(n_epoch)
        train_recon = np.zeros((self.n_arm, n_epoch))
        train_loss_KL = np.zeros((self.n_arm, self.n_categories, n_epoch))
        validation_rec_loss = np.zeros(n_epoch)
        bias_mask = torch.ones(self.n_categories)
        weight_mask = torch.ones((self.n_categories, self.lowD_dim))
        fc_mu = torch.ones((self.state_dim, self.n_categories + self.lowD_dim))
        fc_sigma = torch.ones((self.state_dim, self.n_categories + self.lowD_dim))
        f6_mask = torch.ones((self.lowD_dim, self.state_dim + self.n_categories))

        bias_mask = bias_mask.to(self.device)
        weight_mask = weight_mask.to(self.device)
        fc_mu = fc_mu.to(self.device)
        fc_sigma = fc_sigma.to(self.device)
        f6_mask = f6_mask.to(self.device)
        B = train_loader.batch_size

        if self.init:
            print("Start training ...")
            epoch_time = []
            for epoch in trange(n_epoch):
                train_loss_val = th.zeros(2, device=rank)
                train_jointloss_val = th.zeros(1, device=rank)
                train_dqc = th.zeros(1, device=rank)
                log_dqc = th.zeros(1, device=rank)
                entr = th.zeros(1, device=rank)
                var_min = th.zeros(1, device=rank)
                t0 = time.time()
                train_loss_rec = th.zeros(self.n_arm, device=rank)
                train_KLD_cont = th.zeros(self.n_arm, self.n_categories, device=rank)
                self.model.train()

                train_zcat = [[] for _ in range(self.n_arm)]

                for batch_indx, (data, d_idx), in enumerate(train_loader):
                    data = data.to(self.device)
                    d_idx = d_idx.to(int)
                        
                    tt = time.time() 
                
                    with torch.no_grad():
                        trans_data = self.netA(data.expand(self.n_arm, -1, -1), True, 0.1)[1] if self.aug_file else data.expand(self.n_arm, -1, -1)

                    if self.ref_prior:
                        c_bin = torch.Tensor(c_onehot[d_idx, :]).to(self.device)
                        prior_c = torch.Tensor(c_p[d_idx, :]).to(self.device)
                    else:
                        c_bin = 0.
                        prior_c = 0.

                    self.optimizer.zero_grad()
                    recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, log_qc = self.model(x=trans_data, temp=self.temp, prior_c=prior_c)
                    for arm in range(self.n_arm):
                        train_zcat[arm].append(qc[arm].cpu().data.view(qc[arm].size()[0], self.n_categories).argmax(dim=1).detach().numpy())

                    loss, loss_rec, loss_joint, entropy, dist_c, d_qc, KLD_cont, min_var_0, loglikelihood = \
                        self.model.loss(recon_batch, p_x, r_x, trans_data, mu, log_var, qc, c, c_bin)
                    mem: float = torch.cuda.memory_allocated() / 1e9
                    loss.backward()
                    self.optimizer.step()

                    train_loss_val[0] += loss.data.item()
                    train_loss_val[1] += 1
                    train_jointloss_val += loss_joint
                    train_dqc += d_qc
                    log_dqc += dist_c
                    entr += entropy
                    var_min += min_var_0.data.item()

                    for arm in range(self.n_arm):
                        train_loss_rec[arm] += loss_rec[arm].data.item() / self.input_dim

                num_batches = len(train_loader)
                assert batch_indx + 1 == len(train_loader)
                print('====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {'':.4f}, Distance: {:.4f}, '.format(
                    epoch, train_loss_val[0].data.item() / num_batches, train_loss_rec[0].data.item() / num_batches, 
                     train_dqc.data.item() / num_batches))

                if ws > 1:
                    dist.all_reduce(train_loss_val, op=dist.ReduceOp.SUM)
                    dist.all_reduce(train_dqc, op=dist.ReduceOp.SUM)
                    dist.all_reduce(train_loss_rec, op=dist.ReduceOp.SUM)
                    # dist.all_reduce(train_jointloss_val, op=dist.ReduceOp.SUM)
                    # dist.all_reduce(log_dqc, op=dist.ReduceOp.SUM)
                    # dist.all_reduce(entr, op=dist.ReduceOp.SUM)
                    # dist.all_reduce(log_dqc, op=dist.ReduceOp.SUM)
                    # dist.all_reduce(var_min, op=dist.ReduceOp.SUM)

                train_loss[epoch] = train_loss_val[0] / train_loss_val[1]
                train_loss_joint[epoch] = train_jointloss_val / num_batches
                train_distance[epoch] = train_dqc / train_loss_val[1]
                train_entropy[epoch] = entr / num_batches
                train_log_distance[epoch] = log_dqc / num_batches
                train_minVar[epoch] = var_min / num_batches

                for arm in range(self.n_arm):
                    train_recon[arm, epoch] = train_loss_rec[arm] / train_loss_val[1]
                    for cc in range(self.n_categories):
                        train_loss_KL[arm, cc, epoch] = train_KLD_cont[arm, cc] / num_batches

                _time = time.time() - t0
                print('====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {'':.4f}, Joint Loss: {:.4f}, '
                      'Entropy: {:.4f}, Distance: {:.4f}, Min. Var: {:.6f}, Elapsed Time:{:.2f}, '.format(
                    epoch, train_loss[epoch], train_recon[0, epoch], train_loss_joint[epoch],
                    train_entropy[epoch], train_distance[epoch], train_minVar[epoch], _time))
                
                if run:
                    run.log({
                        'train/total-loss': train_loss[epoch],
                        'train/joint-loss': train_loss_joint[epoch],
                        'train/entropy': train_entropy[epoch],
                        'train/distance': train_distance[epoch],
                        'train/min-var': train_minVar[epoch],
                        'train/time': _time,
                        'train/mem': mem,
                        **dict(map(lambda x: (f'train/rec-loss{x}', train_recon[x, epoch]), range(self.n_arm))),
                    })
                    
                # validation
                self.model.eval()
                with torch.no_grad():
                    val_loss_rec = 0.
                    val_loss = 0.
                    if test_loader.batch_size > 1:
                        for batch_indx, (data_val, d_idx), in enumerate(test_loader):
                            data_val = data_val.to(self.device)
                            d_idx = d_idx.to(int)
                            trans_val_data = []
                            for arm in range(self.n_arm):
                               trans_val_data.append(data_val)

                            if self.ref_prior:
                                c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                            else:
                                c_bin = 0.
                                prior_c = 0.

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c, eval=True)
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data, mu, log_var, qc, c, c_bin)
                            val_loss += loss.data.item()
                            for arm in range(self.n_arm):
                                val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                    else:
                        batch_indx = 0
                        data_val, d_idx = test_loader.dataset.tensors
                        data_val = data_val.to(self.device)
                        d_idx = d_idx.to(int)
                        trans_val_data = []
                        for arm in range(self.n_arm):
                            trans_val_data.append(data_val)

                        if self.ref_prior:
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                        recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data,
                                                                                            temp=self.temp,
                                                                                            prior_c=prior_c, eval=True)
                        loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x,
                                                                                       trans_val_data, mu, log_var, qc,
                                                                                       c, c_bin)
                        val_loss = loss.data.item()
                        for arm in range(self.n_arm):
                            val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                        

                validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.n_arm
                validation_loss[epoch] = val_loss / (batch_indx + 1)
                print('====> Validation Total Loss: {:.4f}, Rec. Loss: {:.4f}'.format(validation_loss[epoch], validation_rec_loss[epoch]))
                if run:
                    run.log({
                        'val/total-loss': validation_loss[epoch],
                        'val/rec-loss': validation_rec_loss[epoch]
                    })

                if self.save and (epoch > 0) and (epoch % 10000 == 0):
                    trained_model = self.folder + f'/model/cpl_mixVAE_model_epoch_{epoch}.pth'
                    torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)

                    
                    predicted_label = np.zeros((self.n_arm, len(train_zcat[0] * B)))
                    for arm in range(self.n_arm):
                        predicted_label[arm] = np.concatenate(train_zcat[arm])

                    # confusion matrix code            
                    c_agreement = []
                    for arm_a in range(self.n_arm):
                        pred_a = predicted_label[arm_a, :]
                        for arm_b in range(arm_a + 1, self.n_arm):
                            pred_b = predicted_label[arm_b, :]
                            armA_vs_armB = np.zeros((self.n_categories, self.n_categories))

                            for samp in range(pred_a.shape[0]):
                                armA_vs_armB[pred_a[samp].astype(int), pred_b[samp].astype(int)] += 1

                            num_samp_arm = []
                            for ij in range(self.n_categories):
                                sum_row = armA_vs_armB[ij, :].sum()
                                sum_column = armA_vs_armB[:, ij].sum()
                                num_samp_arm.append(max(sum_row, sum_column))

                            armA_vs_armB = np.divide(armA_vs_armB, np.array(num_samp_arm), out=np.zeros_like(armA_vs_armB),
                                                    where=np.array(num_samp_arm) != 0)
                            c_agreement.append(np.diag(armA_vs_armB))
                            ind_sort = np.argsort(c_agreement[-1])
                            plt.figure()
                            plt.imshow(armA_vs_armB[:, ind_sort[::-1]][ind_sort[::-1]], cmap='binary')
                            plt.colorbar()
                            plt.xlabel('arm_' + str(arm_a), fontsize=20)
                            plt.xticks(range(self.n_categories), range(self.n_categories))
                            plt.yticks(range(self.n_categories), range(self.n_categories))
                            plt.ylabel('arm_' + str(arm_b), fontsize=20)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title(f'Epoch {epoch} |c|=' + str(self.n_categories), fontsize=20)
                            plt.savefig(self.folder + '/consensus_arm_' + str(arm_a) + '_arm_' + str(arm_b) + '_epoch_' + str(epoch) + '.png', dpi=600)
                            plt.close("all")

                epoch_time.append(time.time() - t0)

            print('epoch time:', np.mean(epoch_time))
            def save_loss_plot(loss_data, label, filename):
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch), loss_data, label=label)
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title(f'{label} loss of the cpl-mixVAE for K={self.n_categories} and S={self.state_dim}')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + f'/model/{filename}_A{self.n_arm}_{self.n_categories}_{self.current_time}.png')
                plt.close()
            
            if self.save and n_epoch > 0:
                # Save train loss plot
                save_loss_plot(train_loss, 'Training', 'train_loss_curve')

                # Save validation loss plot
                save_loss_plot(validation_loss, 'Validation', 'validation_loss_curve')
                
                trained_model = self.folder + f'/model/cpl_mixVAE_model_before_pruning_A{self.n_arm}_' + self.current_time + '.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                bias = self.model.fcc[0].bias.detach().cpu().numpy()
                mask = range(len(bias))
                prune_indx = []
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch), train_loss, label='Training')
                ax.plot(range(n_epoch), validation_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.n_categories) + ' and S=' + str(self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + f'/model/learning_curve_before_pruning_K_A{self.n_arm}_' + str(self.n_categories) + '_' + self.current_time + '.png')
                plt.close("all")

        if n_epoch_p > 0:
            # initialized pruning parameters of the layer of the discrete variable
            bias = self.model.fcc[0].bias.detach().cpu().numpy()
            pruning_mask = np.where(bias != 0.)[0]
            prune_indx = np.where(bias == 0.)[0]
            stop_prune = False
        else:
            stop_prune = True

        pr = self.n_pr
        ind = []
        stop_prune = True
        print('warning: stopping pruning')
        while not stop_prune:
            predicted_label = np.zeros((self.n_arm, len(train_loader.dataset)))

            # Assessment over all dataset
            self.model.eval()
            with torch.no_grad():
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
                        c_bin = 0.
                        prior_c = 0.

                    recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, prior_c, mask=pruning_mask, eval=True)

                    for arm in range(self.n_arm):
                        z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.n_categories).detach().numpy()
                        predicted_label[arm, i * batch_size:min((i + 1) * batch_size, len(train_loader.dataset))] = np.argmax(z_encoder, axis=1)

            c_agreement = []
            for arm_a in range(self.n_arm):
                pred_a = predicted_label[arm_a, :]
                for arm_b in range(arm_a + 1, self.n_arm):
                    pred_b = predicted_label[arm_b, :]
                    armA_vs_armB = np.zeros((self.n_categories, self.n_categories))

                    for samp in range(pred_a.shape[0]):
                        armA_vs_armB[pred_a[samp].astype(int), pred_b[samp].astype(int)] += 1

                    num_samp_arm = []
                    for ij in range(self.n_categories):
                        sum_row = armA_vs_armB[ij, :].sum()
                        sum_column = armA_vs_armB[:, ij].sum()
                        num_samp_arm.append(max(sum_row, sum_column))

                    armA_vs_armB = np.divide(armA_vs_armB, np.array(num_samp_arm), out=np.zeros_like(armA_vs_armB),
                                             where=np.array(num_samp_arm) != 0)
                    c_agreement.append(np.diag(armA_vs_armB))
                    ind_sort = np.argsort(c_agreement[-1])
                    plt.figure()
                    plt.imshow(armA_vs_armB[:, ind_sort[::-1]][ind_sort[::-1]], cmap='binary')
                    plt.colorbar()
                    plt.xlabel('arm_' + str(arm_a), fontsize=20)
                    plt.xticks(range(self.n_categories), range(self.n_categories))
                    plt.yticks(range(self.n_categories), range(self.n_categories))
                    plt.ylabel('arm_' + str(arm_b), fontsize=20)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('|c|=' + str(self.n_categories), fontsize=20)
                    plt.savefig(self.folder + '/consensus_' + str(pr) + '_arm_' + str(arm_a) + '_arm_' + str(arm_b) + '.png', dpi=600)
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
                bias_mask[ind] = 0.
                weight_mask[ind, :] = 0.
                fc_mu[:, self.lowD_dim + ind] = 0.
                fc_sigma[:, self.lowD_dim + ind] = 0.
                f6_mask[:, ind] = 0.
                stop_prune = False
            else:
                print('No more pruning!')
                stop_prune = True

            print("warning: disabled pruning")
            stop_prune = True
            if not stop_prune:
                print("Continue training with pruning ...")
                print(f"Pruned categories: {ind}")
                bias = bias_mask.detach().cpu().numpy()
                pruning_mask = np.where(bias != 0.)[0]
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

                for arm in range(self.n_arm):
                    prune.custom_from_mask(self.model.fcc[arm], 'weight', mask=weight_mask)
                    prune.custom_from_mask(self.model.fcc[arm], 'bias', mask=bias_mask)
                    prune.custom_from_mask(self.model.fc_mu[arm], 'weight', mask=fc_mu)
                    prune.custom_from_mask(self.model.fc_sigma[arm], 'weight', mask=fc_sigma)
                    prune.custom_from_mask(self.model.fc6[arm], 'weight', mask=f6_mask)

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
                    for batch_indx, (data, d_idx), in enumerate(train_loader):
                        # for data in train_loader:
                        data = data.to(self.device)
                        d_idx = d_idx.to(int)
                        data_bin = 0. * data
                        data_bin[data > 0.] = 1.
                        trans_data = []
                        origin_data = []
                        trans_data.append(data)
                        tt = time.time()
                        w_param, bias_param, activ_param = 0, 0, 0
                        # parallelize
                        for arm in range(self.n_arm-1):
                            if self.aug_file:
                                noise = torch.randn(batch_size, self.aug_param['num_n']).to(self.device)
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
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                        self.optimizer.zero_grad()
                        recon_batch, p_x, r_x, x_low, qz, s, z, mu, log_var, log_qz = self.model(trans_data, self.temp, prior_c, mask=pruning_mask)
                        loss, loss_rec, loss_joint, entropy, dist_z, d_qz, KLD_cont, min_var_0, _ = self.model.loss(recon_batch, p_x, r_x,
                                                                                        trans_data, mu, log_var, qz, z, c_bin)

                        loss.backward()
                        self.optimizer.step()
                        ti[batch_indx] = time.time() - tt
                        train_loss_val += loss.data.item()
                        train_jointloss_val += loss_joint
                        train_dqz += d_qz
                        log_dqz += dist_z
                        entr += entropy
                        var_min += min_var_0.data.item()

                        for arm in range(self.n_arm):
                            train_loss_rec[arm] += loss_rec[arm].data.item() / self.input_dim

                    

                    train_loss[epoch] = train_loss_val / (batch_indx + 1)
                    train_loss_joint[epoch] = train_jointloss_val / (batch_indx + 1)
                    train_distance[epoch] = train_dqz / (batch_indx + 1)
                    train_entropy[epoch] = entr / (batch_indx + 1)
                    train_log_distance[epoch] = log_dqz / (batch_indx + 1)
                    train_minVar[epoch] = var_min / (batch_indx + 1)

                    for arm in range(self.n_arm):
                        train_recon[arm, epoch] = train_loss_rec[arm] / (batch_indx + 1)
                        for c in range(self.n_categories):
                            train_loss_KL[arm, c, epoch] = train_KLD_cont[arm, c] / (batch_indx + 1)

                    print('====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {'
                          ':.4f}, Rec_arm_2: {:.4f}, Joint Loss: {:.4f}, Entropy: {:.4f}, Distance: {:.4f}, Elapsed Time:{:.2f}'.format(
                        epoch, train_loss[epoch], train_recon[0, epoch], train_recon[1, epoch], train_loss_joint[epoch],
                        train_entropy[epoch], train_distance[epoch], time.time() - t0))

                    # validation
                    self.model.eval()
                    with torch.no_grad():
                        val_loss_rec = 0.
                        val_loss = 0.
                        if test_loader.batch_size > 1:
                            for batch_indx, (data_val, d_idx), in enumerate(test_loader):
                                d_idx = d_idx.to(int)
                                data_val = data_val.to(self.device)
                                    
                                trans_val_data = []
                                for arm in range(self.n_arm):
                                    trans_val_data.append(data_val)

                                if self.ref_prior:
                                    c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                    prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                                else:
                                    c_bin = 0.
                                    prior_c = 0.

                                recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
                                                                                        eval=True, mask=pruning_mask)
                                loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data,
                                                                                            mu, log_var, qc, c, c_bin)
                                val_loss += loss.data.item()
                                for arm in range(self.n_arm):
                                    val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                        else:
                            batch_indx = 0
                            data_val, d_idx = test_loader.dataset.tensors
                            data_val = data_val.to(self.device)
                            d_idx = d_idx.to(int)
                            trans_val_data = []
                            for arm in range(self.n_arm):
                                trans_val_data.append(data_val)

                            if self.ref_prior:
                                c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                            else:
                                c_bin = 0.
                                prior_c = 0.

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
                                                                                    eval=True, mask=pruning_mask)
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data,
                                                                                        mu, log_var, qc, c, c_bin)
                            val_loss = loss.data.item()
                            for arm in range(self.n_arm):
                                val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                            

                    validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.n_arm
                    total_val_loss[epoch] = val_loss / (batch_indx + 1)
                    print('====> Validation Total Loss: {:.4}, Rec. Loss: {:.4f}'.format(total_val_loss[epoch], validation_rec_loss[epoch]))

                for arm in range(self.n_arm):
                    prune.remove(self.model.fcc[arm], 'weight')
                    prune.remove(self.model.fcc[arm], 'bias')
                    prune.remove(self.model.fc_mu[arm], 'weight')
                    prune.remove(self.model.fc_sigma[arm], 'weight')
                    prune.remove(self.model.fc6[arm], 'weight')


                trained_model = self.folder + '/model/cpl_mixVAE_model_after_pruning_' + str(pr+1) + '_' + self.current_time + '.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch_p), train_loss, label='Training')
                ax.plot(range(n_epoch_p), total_val_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.n_categories) + ' and S=' + str(
                    self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + '../model/learning_curve_after_pruning_' + str(pr+1) + '_K_' + str(
                    self.n_categories) + '_' + self.current_time + '.png')
                plt.close("all")
                pr += 1
        
        print('Training is done!')
    
        # return trained_model
    
    def train_(self, train_loader, test_loader, n_epoch, n_epoch_p, c_p=0, 
               c_onehot=0, min_con=.5, max_prun_it=0, rank=None, world_size=1,
               log=None):

        # assert self.device == rank

        # define current_time
        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # initialized saving arrays
        train_loss = np.zeros(n_epoch)
        validation_loss = np.zeros(n_epoch)
        train_loss_joint = np.zeros(n_epoch)
        train_entropy = np.zeros(n_epoch)
        train_distance = np.zeros(n_epoch)
        train_minVar = np.zeros(n_epoch)
        train_log_distance = np.zeros(n_epoch)
        train_recon = np.zeros((self.n_arm, n_epoch))
        train_loss_KL = np.zeros((self.n_arm, self.n_categories, n_epoch))
        validation_rec_loss = np.zeros(n_epoch)
        bias_mask = torch.ones(self.n_categories)
        weight_mask = torch.ones((self.n_categories, self.lowD_dim))
        fc_mu = torch.ones((self.state_dim, self.n_categories + self.lowD_dim))
        fc_sigma = torch.ones((self.state_dim, self.n_categories + self.lowD_dim))
        f6_mask = torch.ones((self.lowD_dim, self.state_dim + self.n_categories))

        bias_mask = bias_mask.to(rank)
        weight_mask = weight_mask.to(rank)
        fc_mu = fc_mu.to(rank)
        fc_sigma = fc_sigma.to(rank)
        f6_mask = f6_mask.to(rank)
        batch_size = train_loader.batch_size

        if self.init:
            for epoch in range(n_epoch):
                train_loss_val = torch.zeros(2, device=rank)
                train_jointloss_val = 0
                train_dqc = 0
                log_dqc = 0
                entr = 0
                var_min = 0
                t0 = time.time()
                train_loss_rec = np.zeros(self.n_arm)
                train_KLD_cont = np.zeros((self.n_arm, self.n_categories))
                self.model.train()

                sampler = get_sampler(train_loader)
                if is_dist_sampler(sampler):
                    sampler.set_epoch(epoch)

                pbar = mk_pbar(train_loader, mk_rank(rank), enumerate)
                for batch_indx, (data, d_idx), in pbar:
                    # data = to_device(data, rank)
                    # d_idx = map_convert(int, d_idx)
                    data = data.to(rank)
                    d_idx = d_idx.to(int)
                        
                    trans_data = []
                    tt = time.time()
                    for arm in range(self.n_arm):
                        if self.aug_file:
                            noise = torch.randn(batch_size, self.aug_param['num_n'], device=self.device)
                            _, gen_data = self.netA(data, noise, True, rank)
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
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                    if self.ref_prior:
                        c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(rank)
                        prior_c = torch.FloatTensor(c_p[d_idx, :]).to(rank)
                    else:
                        c_bin = 0.
                        prior_c = 0.

                    recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, log_qc = self.model(x=trans_data, temp=self.temp, prior_c=prior_c)
                    loss, loss_rec, loss_joint, entropy, dist_c, d_qc, KLD_cont, min_var_0, loglikelihood = \
                        self.model.loss(recon_batch, p_x, r_x, trans_data, mu, log_var, qc, c, c_bin)
                        
                    self.optimizer.zero_grad()
                    loss.backward()

                    self.optimizer.step()

                    train_loss_val[0] += loss.data.item()
                    train_loss_val[1] += 1
                    train_jointloss_val += loss_joint
                    train_dqc += d_qc
                    log_dqc += dist_c
                    entr += entropy
                    var_min += min_var_0.data.item()

                    for arm in range(self.n_arm):
                        train_loss_rec[arm] += loss_rec[arm].data.item() / self.input_dim

                print(f'before reduce: {train_loss_val[0] / train_loss_val[1]}')
                
                if is_parallel(world_size):
                    dist.all_reduce(train_loss_val, ReduceOp.SUM)

                print(f'after reduce: {train_loss_val[0] / train_loss_val[1]}')                

                batch_count = len(train_loader)
                train_loss[epoch] = train_loss_val[0] / train_loss_val[1]
                train_loss_joint[epoch] = train_jointloss_val / batch_count
                train_distance[epoch] = train_dqc / batch_count
                train_entropy[epoch] = entr / batch_count
                train_log_distance[epoch] = log_dqc / batch_count
                train_minVar[epoch] = var_min / batch_count

                for arm in range(self.n_arm):
                    train_recon[arm, epoch] = train_loss_rec[arm] / batch_count
                    for cc in range(self.n_categories):
                        train_loss_KL[arm, cc, epoch] = train_KLD_cont[arm, cc] / batch_count


                log({'train/recon': train_recon[0, epoch],
                     'train/loss': train_loss[epoch],})

                print_train_loss(epoch, train_loss[epoch], train_recon[0, epoch],
                                train_recon[1, epoch], train_loss_joint[epoch], 
                                train_entropy[epoch], 
                                train_log_distance[epoch], time.time() - t0, rank)

                # validation
                self.model.eval()
                with torch.no_grad():
                    val_loss_rec = 0.
                    val_loss = torch.zeros(1, device=rank)
                    if test_loader.batch_size > 1:
                        pbar = mk_pbar(test_loader, mk_rank(rank), enumerate)
                        for batch_indx, (data_val, d_idx), in pbar:
                            data_val = data_val.to(rank)
                            d_idx = d_idx.to(int)
                            trans_val_data = []
                            for arm in range(self.n_arm):
                               trans_val_data.append(data_val)

                            if self.ref_prior:
                                c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(rank)
                                prior_c = torch.FloatTensor(c_p[d_idx, :]).to(rank)
                            else:
                                c_bin = 0.
                                prior_c = 0.

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c, eval=True)
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data, mu, log_var, qc, c, c_bin)
                            val_loss[0] += loss.data.item()
                            for arm in range(self.n_arm):
                                val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                    else:
                        batch_indx = 0
                        data_val, d_idx = test_loader.dataset.tensors
                        data_val = data_val.to(rank)
                        d_idx = d_idx.to(int)
                        trans_val_data = []
                        for arm in range(self.n_arm):
                            trans_val_data.append(data_val)

                        if self.ref_prior:
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(rank)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(rank)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                        recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data,
                                                                                            temp=self.temp,
                                                                                            prior_c=prior_c, eval=True)
                        loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x,
                                                                                       trans_val_data, mu, log_var, qc,
                                                                                       c, c_bin)
                        val_loss = loss.data.item()
                        for arm in range(self.n_arm):
                            val_loss_rec += loss_rec[arm].data.item() / self.input_dim

                validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.n_arm
                validation_loss[epoch] = val_loss / (batch_indx + 1)

                print_val_loss(validation_loss[epoch], validation_rec_loss[epoch], rank)
                if self.save and (epoch > 0) and (epoch % 1000 == 0):
                    trained_model = self.folder + f'/model/cpl_mixVAE_model_epoch_{epoch}.pth'
                    torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)

            y = avg_recon_loss(train_recon)
            x = range(len(y))
            for e, l in zip(x, y):
                log({'Avg reconstruction loss': l, 'epoch': e})
            self.save = False
            print(f'warning: disabling saving')
            if self.save:
                assert n_epoch > 0, "error: n_epoch must be greater than 0"
                trained_model = self.folder + './model/cpl_mixVAE_model_before_pruning_' + self.current_time + '.pth'
                print(f"trained_model: {trained_model}")
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                bias = self.model.fcc[0].bias.detach().cpu().numpy()
                mask = range(len(bias))
                prune_indx = []
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                # ax.plot(range(n_epoch), train_loss, label='Training')
                print(f"avg_recon_loss(train_recon): {avg_recon_loss(train_recon)}")
                ax.plot(range(n_epoch), avg_recon_loss(train_recon), label='Training')
                # ax.plot(range(n_epoch), validation_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.n_categories) + ' and S=' + str(self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + '/model/learning_curve_before_pruning_K_' + str(self.n_categories) + '_' + self.current_time + '.png')
                plt.close("all")

        if n_epoch_p > 0:
            # initialized pruning parameters of the layer of the discrete variable
            bias = self.model.fcc[0].bias.detach().cpu().numpy()
            pruning_mask = np.where(bias != 0.)[0]
            prune_indx = np.where(bias == 0.)[0]
            stop_prune = False
        else:
            stop_prune = True

        pr = self.n_pr
        ind = []
        if is_master(rank):
            print("warning: disabled pruning")
        stop_prune = True
        n_epoch_p = 0
        while not stop_prune:
            predicted_label = np.zeros((self.n_arm, len(train_loader.dataset)))

            # Assessment over all dataset
            self.model.eval()
            with torch.no_grad():
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
                        c_bin = 0.
                        prior_c = 0.

                    recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, prior_c, mask=pruning_mask, eval=True)

                    for arm in range(self.n_arm):
                        z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.n_categories).detach().numpy()
                        predicted_label[arm, i * batch_size:min((i + 1) * batch_size, len(train_loader.dataset))] = np.argmax(z_encoder, axis=1)

            c_agreement = []
            for arm_a in range(self.n_arm):
                pred_a = predicted_label[arm_a, :]
                for arm_b in range(arm_a + 1, self.n_arm):
                    pred_b = predicted_label[arm_b, :]
                    armA_vs_armB = np.zeros((self.n_categories, self.n_categories))

                    for samp in range(pred_a.shape[0]):
                        armA_vs_armB[pred_a[samp].astype(int), pred_b[samp].astype(int)] += 1

                    num_samp_arm = []
                    for ij in range(self.n_categories):
                        sum_row = armA_vs_armB[ij, :].sum()
                        sum_column = armA_vs_armB[:, ij].sum()
                        num_samp_arm.append(max(sum_row, sum_column))

                    armA_vs_armB = np.divide(armA_vs_armB, np.array(num_samp_arm), out=np.zeros_like(armA_vs_armB),
                                             where=np.array(num_samp_arm) != 0)
                    c_agreement.append(np.diag(armA_vs_armB))
                    ind_sort = np.argsort(c_agreement[-1])
                    plt.figure()
                    plt.imshow(armA_vs_armB[:, ind_sort[::-1]][ind_sort[::-1]], cmap='binary')
                    plt.colorbar()
                    plt.xlabel('arm_' + str(arm_a), fontsize=20)
                    plt.xticks(range(self.n_categories), range(self.n_categories))
                    plt.yticks(range(self.n_categories), range(self.n_categories))
                    plt.ylabel('arm_' + str(arm_b), fontsize=20)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('|c|=' + str(self.n_categories), fontsize=20)
                    plt.savefig(self.folder + '/consensus_' + str(pr) + '_arm_' + str(arm_a) + '_arm_' + str(arm_b) + '.png', dpi=600)
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
                bias_mask[ind] = 0.
                weight_mask[ind, :] = 0.
                fc_mu[:, self.lowD_dim + ind] = 0.
                fc_sigma[:, self.lowD_dim + ind] = 0.
                f6_mask[:, ind] = 0.
                stop_prune = False
            else:
                print('No more pruning!')
                stop_prune = True
            if not stop_prune:
                print("Continue training with pruning ...")
                print(f"Pruned categories: {ind}")
                bias = bias_mask.detach().cpu().numpy()
                pruning_mask = np.where(bias != 0.)[0]
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

                for arm in range(self.n_arm):
                    prune.custom_from_mask(self.model.fcc[arm], 'weight', mask=weight_mask)
                    prune.custom_from_mask(self.model.fcc[arm], 'bias', mask=bias_mask)
                    prune.custom_from_mask(self.model.fc_mu[arm], 'weight', mask=fc_mu)
                    prune.custom_from_mask(self.model.fc_sigma[arm], 'weight', mask=fc_sigma)
                    prune.custom_from_mask(self.model.fc6[arm], 'weight', mask=f6_mask)

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
                    for batch_indx, (data, d_idx), in enumerate(train_loader):
                        # for data in train_loader:
                        data = data.to(self.device)
                        d_idx = d_idx.to(int)
                        data_bin = 0. * data
                        data_bin[data > 0.] = 1.
                        trans_data = []
                        origin_data = []
                        trans_data.append(data)
                        tt = time.time()
                        w_param, bias_param, activ_param = 0, 0, 0
                        for arm in range(self.n_arm-1):
                            if self.aug_file:
                                noise = torch.randn(batch_size, self.aug_param['num_n']).to(self.device)
                                _, gen_data = self.netA(data, noise, True, self.device)
                                if self.aug_param['n_zim'] > 1:
                                    data_bin = 0. * data
                                    data_bin[data > self.eps] = 1.
                                    fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                                    trans_data.append(fake_data)
                                else:
                                    trans_data.append(gen_data)
                            else:
                                trans_data.append(data)

                        if self.ref_prior:
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                        self.optimizer.zero_grad()
                        recon_batch, p_x, r_x, x_low, qz, s, z, mu, log_var, log_qz = self.model(trans_data, self.temp, prior_c, mask=pruning_mask)
                        loss, loss_rec, loss_joint, entropy, dist_z, d_qz, KLD_cont, min_var_0, _ = self.model.loss(recon_batch, p_x, r_x,
                                                                                        trans_data, mu, log_var, qz, z, c_bin)

                        loss.backward()
                        self.optimizer.step()
                        ti[batch_indx] = time.time() - tt
                        train_loss_val += loss.data.item()
                        train_jointloss_val += loss_joint
                        train_dqz += d_qz
                        log_dqz += dist_z
                        entr += entropy
                        var_min += min_var_0.data.item()

                        for arm in range(self.n_arm):
                            train_loss_rec[arm] += loss_rec[arm].data.item() / self.input_dim

                    train_loss[epoch] = train_loss_val / (batch_indx + 1)
                    train_loss_joint[epoch] = train_jointloss_val / (batch_indx + 1)
                    train_distance[epoch] = train_dqz / (batch_indx + 1)
                    train_entropy[epoch] = entr / (batch_indx + 1)
                    train_log_distance[epoch] = log_dqz / (batch_indx + 1)
                    train_minVar[epoch] = var_min / (batch_indx + 1)

                    for arm in range(self.n_arm):
                        train_recon[arm, epoch] = train_loss_rec[arm] / (batch_indx + 1)
                        for c in range(self.n_categories):
                            train_loss_KL[arm, c, epoch] = train_KLD_cont[arm, c] / (batch_indx + 1)

                    print('====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {'
                          ':.4f}, Rec_arm_2: {:.4f}, Joint Loss: {:.4f}, Entropy: {:.4f}, Distance: {:.4f}, Elapsed Time:{:.2f}'.format(
                        epoch, train_loss[epoch], train_recon[0, epoch], train_recon[1, epoch], train_loss_joint[epoch],
                        train_entropy[epoch], train_distance[epoch], time.time() - t0))

                    # validation
                    self.model.eval()
                    with torch.no_grad():
                        val_loss_rec = 0.
                        val_loss = 0.
                        if test_loader.batch_size > 1:
                            for batch_indx, (data_val, d_idx), in enumerate(test_loader):
                                d_idx = d_idx.to(int)
                                data_val = data_val.to(self.device)
                                    
                                trans_val_data = []
                                for arm in range(self.n_arm):
                                    trans_val_data.append(data_val)

                                if self.ref_prior:
                                    c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                    prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                                else:
                                    c_bin = 0.
                                    prior_c = 0.

                                recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
                                                                                        eval=True, mask=pruning_mask)
                                loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data,
                                                                                            mu, log_var, qc, c, c_bin)
                                val_loss += loss.data.item()
                                for arm in range(self.n_arm):
                                    val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                        else:
                            batch_indx = 0
                            data_val, d_idx = test_loader.dataset.tensors
                            data_val = data_val.to(self.device)
                            d_idx = d_idx.to(int)
                            trans_val_data = []
                            for arm in range(self.n_arm):
                                trans_val_data.append(data_val)

                            if self.ref_prior:
                                c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                            else:
                                c_bin = 0.
                                prior_c = 0.

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
                                                                                    eval=True, mask=pruning_mask)
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data,
                                                                                        mu, log_var, qc, c, c_bin)
                            val_loss = loss.data.item()
                            for arm in range(self.n_arm):
                                val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                            

                    validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.n_arm
                    total_val_loss[epoch] = val_loss / (batch_indx + 1)
                    print('====> Validation Total Loss: {:.4}, Rec. Loss: {:.4f}'.format(total_val_loss[epoch], validation_rec_loss[epoch]))

                for arm in range(self.n_arm):
                    prune.remove(self.model.fcc[arm], 'weight')
                    prune.remove(self.model.fcc[arm], 'bias')
                    prune.remove(self.model.fc_mu[arm], 'weight')
                    prune.remove(self.model.fc_sigma[arm], 'weight')
                    prune.remove(self.model.fc6[arm], 'weight')

                trained_model = self.folder + '/model/cpl_mixVAE_model_after_pruning_' + str(pr+1) + '_' + self.current_time + '.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch_p), train_loss, label='Training')
                ax.plot(range(n_epoch_p), total_val_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.n_categories) + ' and S=' + str(
                    self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + '/model/learning_curve_after_pruning_' + str(pr+1) + '_K_' + str(
                    self.n_categories) + '_' + self.current_time + '.png')
                plt.close("all")
                pr += 1
    
        return ''
    
    def train_joint_(self, train_loader, test_loader, n_epoch, n_epoch_p, c_p=0, 
               c_onehot=0, min_con=.5, max_prun_it=0, rank=None, world_size=1,
               log=None):
        assert th.cuda.is_available(), "error: no GPU available"

        # define current_time
        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # initialized saving arrays
        train_loss = np.zeros(n_epoch)
        validation_loss = np.zeros(n_epoch)
        train_loss_joint = np.zeros(n_epoch)
        train_entropy = np.zeros(n_epoch)
        train_distance = np.zeros(n_epoch)
        train_minVar = np.zeros(n_epoch)
        train_log_distance = np.zeros(n_epoch)
        train_recon = np.zeros((self.n_arm, n_epoch))
        train_loss_KL = np.zeros((self.n_arm, self.n_categories, n_epoch))
        validation_rec_loss = np.zeros(n_epoch)
        bias_mask = torch.ones(self.n_categories)
        weight_mask = torch.ones((self.n_categories, self.lowD_dim))
        fc_mu = torch.ones((self.state_dim, self.n_categories + self.lowD_dim))
        fc_sigma = torch.ones((self.state_dim, self.n_categories + self.lowD_dim))
        f6_mask = torch.ones((self.lowD_dim, self.state_dim + self.n_categories))

        bias_mask = bias_mask.to(rank)
        weight_mask = weight_mask.to(rank)
        fc_mu = fc_mu.to(rank)
        fc_sigma = fc_sigma.to(rank)
        f6_mask = f6_mask.to(rank)
        batch_size = train_loader.batch_size

        netD = Discriminator(input_dim=self.netA.input_dim).to(rank)
        # TODO: make these functional
        criterionD = nn.BCELoss()
        mseDist = nn.MSELoss()

        optimD = optim.Adam([{'params': netD.parameters()}], lr=self.aug_param['learning_rate'])
        optimA = optim.Adam([{'params': self.netA.parameters()}], lr=self.aug_parm['learning_rate'])

        REAL_LABEL = 1.
        FAKE_LABEL = 0.
        A_losses = []
        D_losses = []

        if self.init:
            for epoch in range(n_epoch):
                train_loss_val = torch.zeros(2, device=rank)
                train_jointloss_val = 0
                train_dqc = 0
                log_dqc = 0
                entr = 0
                var_min = 0
                t0 = time.time()
                train_loss_rec = np.zeros(self.n_arm)
                train_KLD_cont = np.zeros((self.n_arm, self.n_categories))
                self.model.train()

                # vae gan losses
                A_loss_e, D_loss_e = 0, 0
                gen_loss_e, recon_loss_e = 0, 0
                triplet_loss_e = 0
                n_adv = 0

                sampler = get_sampler(train_loader)
                if is_dist_sampler(sampler):
                    sampler.set_epoch(epoch)

                pbar = mk_pbar(train_loader, mk_rank(rank), enumerate)
                for batch_indx, (data, d_idx), in pbar:
                    data = data.to(rank)
                    data_bin = torch.where(data > self.eps, 1, 0).to(rank)
                    d_idx = d_idx.to(int)
                        
                    optimD.zero_grad()
                    label = torch.full((batch_size,), REAL_LABEL, device=rank)
                    _, probs_real = netD(data_bin)
                    loss_real = criterionD(probs_real.view(-1), label)

                    if F.relu(loss_real - np.log(2) / 2) > 0:
                        loss_real.backward()
                        optim_D = True
                    else:
                        optim_D = False

                    label.fill_(FAKE_LABEL)

                    trans_data = []
                    tt = time.time()
                    for arm in range(self.n_arm):
                        if self.aug_file:
                            noise = torch.randn(batch_size, self.aug_param['num_n'], device=rank)
                            _, gen_data = self.netA(data, noise, True, rank)
                            _, gen_data2 = self.netA(data, noise, False, rank)
                            # if self.aug_param['n_zim'] > 1:
                            #     data_bin = 0. * data
                            #     data_bin[data > self.eps] = 1.
                            #     fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                            #     trans_data.append(fake_data)
                            # else:
                            trans_data.append(gen_data)

                            gen_data_bin = 0. * gen_data
                            gen_data2_bin = 0. * gen_data2
                            gen_data_bin[gen_data > 1e-3] = 1.
                            gen_data2_bin[gen_data2 > 1e-3] = 1.
                            gen_data = 1. * gen_data2

                        else:
                            trans_data.append(data)

                        if self.ref_prior:
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                    if self.ref_prior:
                        c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(rank)
                        prior_c = torch.FloatTensor(c_p[d_idx, :]).to(rank)
                    else:
                        c_bin = 0.
                        prior_c = 0.

                    recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, log_qc = self.model(x=trans_data, temp=self.temp, prior_c=prior_c)
                    loss, loss_rec, loss_joint, entropy, dist_c, d_qc, KLD_cont, min_var_0, loglikelihood = \
                        self.model.loss(recon_batch, p_x, r_x, trans_data, mu, log_var, qc, c, c_bin)
                        
                    self.optimizer.zero_grad()
                    loss.backward()

                    self.optimizer.step()

                    train_loss_val[0] += loss.data.item()
                    train_loss_val[1] += 1
                    train_jointloss_val += loss_joint
                    train_dqc += d_qc
                    log_dqc += dist_c
                    entr += entropy
                    var_min += min_var_0.data.item()

                    for arm in range(self.n_arm):
                        train_loss_rec[arm] += loss_rec[arm].data.item() / self.input_dim
                
                if is_parallel(world_size):
                    dist.all_reduce(train_loss_val, ReduceOp.SUM)

                batch_count = len(train_loader)
                train_loss[epoch] = train_loss_val[0] / train_loss_val[1]
                train_loss_joint[epoch] = train_jointloss_val / batch_count
                train_distance[epoch] = train_dqc / batch_count
                train_entropy[epoch] = entr / batch_count
                train_log_distance[epoch] = log_dqc / batch_count
                train_minVar[epoch] = var_min / batch_count

                for arm in range(self.n_arm):
                    train_recon[arm, epoch] = train_loss_rec[arm] / batch_count
                    for cc in range(self.n_categories):
                        train_loss_KL[arm, cc, epoch] = train_KLD_cont[arm, cc] / batch_count

                log({'train_recon': train_recon[0, epoch]})

                print_train_loss(epoch, train_loss[epoch], train_recon[0, epoch],
                                train_recon[1, epoch], train_loss_joint[epoch], 
                                train_entropy[epoch], 
                                train_log_distance[epoch], time.time() - t0, rank)

                # validation
                self.model.eval()
                with torch.no_grad():
                    val_loss_rec = 0.
                    val_loss = torch.zeros(1, device=rank)
                    if test_loader.batch_size > 1:
                        pbar = mk_pbar(test_loader, mk_rank(rank), enumerate)
                        for batch_indx, (data_val, d_idx), in pbar:
                            data_val = data_val.to(rank)
                            d_idx = d_idx.to(int)
                            trans_val_data = []
                            for arm in range(self.n_arm):
                               trans_val_data.append(data_val)

                            if self.ref_prior:
                                c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(rank)
                                prior_c = torch.FloatTensor(c_p[d_idx, :]).to(rank)
                            else:
                                c_bin = 0.
                                prior_c = 0.

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c, eval=True)
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data, mu, log_var, qc, c, c_bin)
                            val_loss[0] += loss.data.item()
                            for arm in range(self.n_arm):
                                val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                    else:
                        batch_indx = 0
                        data_val, d_idx = test_loader.dataset.tensors
                        data_val = data_val.to(rank)
                        d_idx = d_idx.to(int)
                        trans_val_data = []
                        for arm in range(self.n_arm):
                            trans_val_data.append(data_val)

                        if self.ref_prior:
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(rank)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(rank)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                        recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data,
                                                                                            temp=self.temp,
                                                                                            prior_c=prior_c, eval=True)
                        loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x,
                                                                                       trans_val_data, mu, log_var, qc,
                                                                                       c, c_bin)
                        val_loss = loss.data.item()
                        for arm in range(self.n_arm):
                            val_loss_rec += loss_rec[arm].data.item() / self.input_dim

                validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.n_arm
                validation_loss[epoch] = val_loss / (batch_indx + 1)

                print_val_loss(validation_loss[epoch], validation_rec_loss[epoch], rank)
                if self.save and (epoch > 0) and (epoch % 1000 == 0):
                    trained_model = self.folder + f'/model/cpl_mixVAE_model_epoch_{epoch}.pth'
                    torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)

            y = avg_recon_loss(train_recon)
            x = range(len(y))
            for e, l in zip(x, y):
                log({'Avg reconstruction loss': l, 'epoch': e})
            self.save = False
            prn(f'warning: disabling saving')
            if self.save:
                assert n_epoch > 0, "error: n_epoch must be greater than 0"
                trained_model = self.folder + './model/cpl_mixVAE_model_before_pruning_' + self.current_time + '.pth'
                print(f"trained_model: {trained_model}")
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                bias = self.model.fcc[0].bias.detach().cpu().numpy()
                mask = range(len(bias))
                prune_indx = []
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                # ax.plot(range(n_epoch), train_loss, label='Training')
                print(f"avg_recon_loss(train_recon): {avg_recon_loss(train_recon)}")
                ax.plot(range(n_epoch), avg_recon_loss(train_recon), label='Training')
                # ax.plot(range(n_epoch), validation_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.n_categories) + ' and S=' + str(self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + '/model/learning_curve_before_pruning_K_' + str(self.n_categories) + '_' + self.current_time + '.png')
                plt.close("all")

        if n_epoch_p > 0:
            # initialized pruning parameters of the layer of the discrete variable
            bias = self.model.fcc[0].bias.detach().cpu().numpy()
            pruning_mask = np.where(bias != 0.)[0]
            prune_indx = np.where(bias == 0.)[0]
            stop_prune = False
        else:
            stop_prune = True

        pr = self.n_pr
        ind = []
        if is_master(rank):
            print("warning: disabled pruning")
        stop_prune = True
        n_epoch_p = 0
        while not stop_prune:
            predicted_label = np.zeros((self.n_arm, len(train_loader.dataset)))

            # Assessment over all dataset
            self.model.eval()
            with torch.no_grad():
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
                        c_bin = 0.
                        prior_c = 0.

                    recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, prior_c, mask=pruning_mask, eval=True)

                    for arm in range(self.n_arm):
                        z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.n_categories).detach().numpy()
                        predicted_label[arm, i * batch_size:min((i + 1) * batch_size, len(train_loader.dataset))] = np.argmax(z_encoder, axis=1)

            c_agreement = []
            for arm_a in range(self.n_arm):
                pred_a = predicted_label[arm_a, :]
                for arm_b in range(arm_a + 1, self.n_arm):
                    pred_b = predicted_label[arm_b, :]
                    armA_vs_armB = np.zeros((self.n_categories, self.n_categories))

                    for samp in range(pred_a.shape[0]):
                        armA_vs_armB[pred_a[samp].astype(int), pred_b[samp].astype(int)] += 1

                    num_samp_arm = []
                    for ij in range(self.n_categories):
                        sum_row = armA_vs_armB[ij, :].sum()
                        sum_column = armA_vs_armB[:, ij].sum()
                        num_samp_arm.append(max(sum_row, sum_column))

                    armA_vs_armB = np.divide(armA_vs_armB, np.array(num_samp_arm), out=np.zeros_like(armA_vs_armB),
                                             where=np.array(num_samp_arm) != 0)
                    c_agreement.append(np.diag(armA_vs_armB))
                    ind_sort = np.argsort(c_agreement[-1])
                    plt.figure()
                    plt.imshow(armA_vs_armB[:, ind_sort[::-1]][ind_sort[::-1]], cmap='binary')
                    plt.colorbar()
                    plt.xlabel('arm_' + str(arm_a), fontsize=20)
                    plt.xticks(range(self.n_categories), range(self.n_categories))
                    plt.yticks(range(self.n_categories), range(self.n_categories))
                    plt.ylabel('arm_' + str(arm_b), fontsize=20)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('|c|=' + str(self.n_categories), fontsize=20)
                    plt.savefig(self.folder + '/consensus_' + str(pr) + '_arm_' + str(arm_a) + '_arm_' + str(arm_b) + '.png', dpi=600)
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
                bias_mask[ind] = 0.
                weight_mask[ind, :] = 0.
                fc_mu[:, self.lowD_dim + ind] = 0.
                fc_sigma[:, self.lowD_dim + ind] = 0.
                f6_mask[:, ind] = 0.
                stop_prune = False
            else:
                print('No more pruning!')
                stop_prune = True
            if not stop_prune:
                print("Continue training with pruning ...")
                print(f"Pruned categories: {ind}")
                bias = bias_mask.detach().cpu().numpy()
                pruning_mask = np.where(bias != 0.)[0]
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

                for arm in range(self.n_arm):
                    prune.custom_from_mask(self.model.fcc[arm], 'weight', mask=weight_mask)
                    prune.custom_from_mask(self.model.fcc[arm], 'bias', mask=bias_mask)
                    prune.custom_from_mask(self.model.fc_mu[arm], 'weight', mask=fc_mu)
                    prune.custom_from_mask(self.model.fc_sigma[arm], 'weight', mask=fc_sigma)
                    prune.custom_from_mask(self.model.fc6[arm], 'weight', mask=f6_mask)

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
                    for batch_indx, (data, d_idx), in enumerate(train_loader):
                        # for data in train_loader:
                        data = data.to(self.device)
                        d_idx = d_idx.to(int)
                        data_bin = 0. * data
                        data_bin[data > 0.] = 1.
                        trans_data = []
                        origin_data = []
                        trans_data.append(data)
                        tt = time.time()
                        w_param, bias_param, activ_param = 0, 0, 0
                        for arm in range(self.n_arm-1):
                            if self.aug_file:
                                noise = torch.randn(batch_size, self.aug_param['num_n']).to(self.device)
                                _, gen_data = self.netA(data, noise, True, self.device)
                                if self.aug_param['n_zim'] > 1:
                                    data_bin = 0. * data
                                    data_bin[data > self.eps] = 1.
                                    fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                                    trans_data.append(fake_data)
                                else:
                                    trans_data.append(gen_data)
                            else:
                                trans_data.append(data)

                        if self.ref_prior:
                            c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                            prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                        else:
                            c_bin = 0.
                            prior_c = 0.

                        self.optimizer.zero_grad()
                        recon_batch, p_x, r_x, x_low, qz, s, z, mu, log_var, log_qz = self.model(trans_data, self.temp, prior_c, mask=pruning_mask)
                        loss, loss_rec, loss_joint, entropy, dist_z, d_qz, KLD_cont, min_var_0, _ = self.model.loss(recon_batch, p_x, r_x,
                                                                                        trans_data, mu, log_var, qz, z, c_bin)

                        loss.backward()
                        self.optimizer.step()
                        ti[batch_indx] = time.time() - tt
                        train_loss_val += loss.data.item()
                        train_jointloss_val += loss_joint
                        train_dqz += d_qz
                        log_dqz += dist_z
                        entr += entropy
                        var_min += min_var_0.data.item()

                        for arm in range(self.n_arm):
                            train_loss_rec[arm] += loss_rec[arm].data.item() / self.input_dim

                    train_loss[epoch] = train_loss_val / (batch_indx + 1)
                    train_loss_joint[epoch] = train_jointloss_val / (batch_indx + 1)
                    train_distance[epoch] = train_dqz / (batch_indx + 1)
                    train_entropy[epoch] = entr / (batch_indx + 1)
                    train_log_distance[epoch] = log_dqz / (batch_indx + 1)
                    train_minVar[epoch] = var_min / (batch_indx + 1)

                    for arm in range(self.n_arm):
                        train_recon[arm, epoch] = train_loss_rec[arm] / (batch_indx + 1)
                        for c in range(self.n_categories):
                            train_loss_KL[arm, c, epoch] = train_KLD_cont[arm, c] / (batch_indx + 1)

                    print('====> Epoch:{}, Total Loss: {:.4f}, Rec_arm_1: {'
                          ':.4f}, Rec_arm_2: {:.4f}, Joint Loss: {:.4f}, Entropy: {:.4f}, Distance: {:.4f}, Elapsed Time:{:.2f}'.format(
                        epoch, train_loss[epoch], train_recon[0, epoch], train_recon[1, epoch], train_loss_joint[epoch],
                        train_entropy[epoch], train_distance[epoch], time.time() - t0))

                    # validation
                    self.model.eval()
                    with torch.no_grad():
                        val_loss_rec = 0.
                        val_loss = 0.
                        if test_loader.batch_size > 1:
                            for batch_indx, (data_val, d_idx), in enumerate(test_loader):
                                d_idx = d_idx.to(int)
                                data_val = data_val.to(self.device)
                                    
                                trans_val_data = []
                                for arm in range(self.n_arm):
                                    trans_val_data.append(data_val)

                                if self.ref_prior:
                                    c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                    prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                                else:
                                    c_bin = 0.
                                    prior_c = 0.

                                recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
                                                                                        eval=True, mask=pruning_mask)
                                loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data,
                                                                                            mu, log_var, qc, c, c_bin)
                                val_loss += loss.data.item()
                                for arm in range(self.n_arm):
                                    val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                        else:
                            batch_indx = 0
                            data_val, d_idx = test_loader.dataset.tensors
                            data_val = data_val.to(self.device)
                            d_idx = d_idx.to(int)
                            trans_val_data = []
                            for arm in range(self.n_arm):
                                trans_val_data.append(data_val)

                            if self.ref_prior:
                                c_bin = torch.FloatTensor(c_onehot[d_idx, :]).to(self.device)
                                prior_c = torch.FloatTensor(c_p[d_idx, :]).to(self.device)
                            else:
                                c_bin = 0.
                                prior_c = 0.

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, prior_c=prior_c,
                                                                                    eval=True, mask=pruning_mask)
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data,
                                                                                        mu, log_var, qc, c, c_bin)
                            val_loss = loss.data.item()
                            for arm in range(self.n_arm):
                                val_loss_rec += loss_rec[arm].data.item() / self.input_dim
                            

                    validation_rec_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.n_arm
                    total_val_loss[epoch] = val_loss / (batch_indx + 1)
                    print('====> Validation Total Loss: {:.4}, Rec. Loss: {:.4f}'.format(total_val_loss[epoch], validation_rec_loss[epoch]))

                for arm in range(self.n_arm):
                    prune.remove(self.model.fcc[arm], 'weight')
                    prune.remove(self.model.fcc[arm], 'bias')
                    prune.remove(self.model.fc_mu[arm], 'weight')
                    prune.remove(self.model.fc_sigma[arm], 'weight')
                    prune.remove(self.model.fc6[arm], 'weight')

                trained_model = self.folder + '/model/cpl_mixVAE_model_after_pruning_' + str(pr+1) + '_' + self.current_time + '.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch_p), train_loss, label='Training')
                ax.plot(range(n_epoch_p), total_val_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.n_categories) + ' and S=' + str(
                    self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + '/model/learning_curve_after_pruning_' + str(pr+1) + '_K_' + str(
                    self.n_categories) + '_' + self.current_time + '.png')
                plt.close("all")
                pr += 1
    
        return ''

    def eval_model(self, ldr: DataLoader, c_p=0, c_onehot=0):
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

        # Set the model to evaluation mode
        self.model.eval()

        # Extract bias and pruning mask
        bias = asnp(self.model.fcc[0].bias)
        pruning_mask = np.where(bias != 0.)[0]
        prune_indx = np.where(bias == 0.)[0]

        # Initialize arrays for storing evaluation results
        N = len(ldr.dataset)
        recon_cell = np.zeros((A, N, self.input_dim))
        p_cell = np.zeros((A, N, self.input_dim))
        state_sample = np.zeros((A, N, self.state_dim))
        state_mu = np.zeros((A, N, self.state_dim))
        state_var = np.zeros((A, N, self.state_dim))
        z_prob = np.zeros((A, N, C))
        z_sample = np.zeros((A, N, C))
        data_low = np.zeros((A, N, self.lowD_dim))
        state_cat = np.zeros([A, N])
        prob_cat = np.zeros([A, N])
        if self.ref_prior:
            predicted_label = np.zeros((A + 1, N))
        else:
            predicted_label = np.zeros((A, N))
        data_indx = np.zeros(N)
        total_loss_val = []
        total_dist_z = []
        total_dist_qz = []
        total_loss_rec = [[] for _ in range(A)]
        total_loglikelihood = [[] for _ in range(A)]

        # Perform evaluation
        self.model.eval()
        B = unwrap(ldr.batch_size)

        with th.no_grad():
            if B > 1:
                for i, (data, data_idx) in enumerate(ldr):
                    data = data.to(self.device)
                    data_idx = data_idx.to(int)

                    start = i * B
                    end = min((i + 1) * B, N)
                    
                    if self.ref_prior:
                        c_bin = th.FloatTensor(c_onehot[data_idx, :]).to(self.device)
                        prior_c = th.FloatTensor(c_p[data_idx, :]).to(self.device)
                    else:
                        c_bin = 0.
                        prior_c = 0.

                    trans_data = [data for _ in range(A)]

                    recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, prior_c=prior_c, eval=True, mask=pruning_mask)
                    loss, loss_arms, _, _, dist_z, d_qz, _, _, loglikelihood = self.model.loss(recon, p_x, r_x, trans_data, mu, log_sigma, z_category, z_smp, c_bin)
                    total_loss_val.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
                    total_dist_z.append(dist_z.item() if isinstance(dist_z, torch.Tensor) else dist_z)
                    total_dist_qz.append(d_qz.item() if isinstance(d_qz, torch.Tensor) else d_qz)

                    if self.ref_prior:
                        predicted_label[0, start:end] = np.argmax(c_p[data_idx, :], axis=1) + 1

                    for a in range(A):
                        total_loss_rec[a].append(loss_arms[a].item())
                        total_loglikelihood[a].append(loglikelihood[a].item())

                    for a in range(A):
                        state_sample[a, start:end, :] = asnp(state[a])
                        state_mu[a, start:end, :] = asnp(mu[a])
                        state_var[a, start:end, :] = asnp(log_sigma[a])
                        assert z_category[a].size()[0] == len(z_category[a])
                        z_encoder = asnp(z_category[a].view(len(z_category[a]), C))
                        z_prob[a, start:end, :] = z_encoder
                        z_samp = asnp(z_smp[a].view(len(z_smp[a]), C))
                        z_sample[a, start:end, :] = z_samp
                        data_low[a,  start:end, :] = asnp(x_low[a])
                        label = data_idx.numpy().astype(int)
                        data_indx[start:end] = label
                        recon_cell[a, start:end, :] = asnp(recon[a])

                        assert z_encoder.shape[0] == len(z_encoder)
                        for n in range(len(z_encoder)):
                            state_cat[a, start + n] = np.argmax(z_encoder[n, :]) + 1
                            prob_cat[a, start + n] = np.max(z_encoder[n, :])

                        if self.ref_prior:
                            predicted_label[a+1, start:end] = np.argmax(z_encoder, axis=1) + 1
                        else:
                            predicted_label[a, start:end] = np.argmax(z_encoder, axis=1) + 1

            else:
                i = 0
                data, data_idx = ldr.dataset.tensors
                data = data.to(self.device)
                data_idx = data_idx.to(int)
                if self.ref_prior:
                    c_bin = th.FloatTensor(c_onehot[data_idx, :]).to(self.device)
                    prior_c = th.FloatTensor(c_p[data_idx, :]).to(self.device)
                else:
                    c_bin = 0.
                    prior_c = 0.

                trans_data = [data for _ in range(A)]

                recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, prior_c=prior_c, eval=True, mask=pruning_mask)
                loss, loss_arms, loss_joint, _, dist_z, d_qz, _, _, loglikelihood = self.model.loss(recon, p_x, r_x, trans_data, mu, log_sigma, z_category, z_smp, c_bin)
                total_loss_val = loss.item()
                total_dist_z = dist_z.item()
                total_dist_qz = d_qz.item()
                if self.ref_prior:
                    predicted_label[0, :] = np.argmax(c_p[data_idx, :], axis=1) + 1

                for a in range(A):
                    total_loss_rec[a] = loss_arms[a].item()
                    total_loglikelihood[a] = loglikelihood[a].item()

                for a in range(A):
                    state_sample[a, :, :] = asnp(state[a])
                    state_mu[a, :, :] = asnp(mu[a])
                    state_var[a, :, :] = asnp(log_sigma[a])
                    assert z_category[a].size()[0] == len(z_category[a])
                    z_encoder = asnp(z_category.view(len(z_category[a]), C)) # TODO: check
                    z_prob[a, :, :] = z_encoder
                    assert z_smp[a].size()[0] == len(z_smp[a])
                    z_samp = asnp(z_smp[a].view(len(z_smp[a]), C)) # TODO: check
                    z_sample[a, :, :] = z_samp
                    data_low[a, :, :] = asnp(x_low[a])
                    label = data_idx.numpy().astype(int)
                    data_indx = label
                    recon_cell[a, :, :] = asnp(recon[a])

                    for n in range(z_encoder.shape[0]):
                        state_cat[a, n] = np.argmax(z_encoder[n, :]) + 1
                        prob_cat[a, n] = np.max(z_encoder[n, :])

                    if self.ref_prior:
                        predicted_label[a + 1, :] = np.argmax(z_encoder, axis=1) + 1
                    else:
                        predicted_label[a, ] = np.argmax(z_encoder, axis=1) + 1

        return {
            'state_sample': state_sample,
            'state_mu': state_mu,
            'state_var': state_var,
            'state_cat': state_cat,
            'prob_cat': prob_cat,
            'total_loss_rec': np.array([np.mean(np.array(total_loss_rec[a])) for a in range(A)]),
            'total_likelihood': np.array([np.mean(np.array(total_loglikelihood[a])) for a in range(A)]),
            'total_dist_z': np.mean(np.array(total_dist_z)),
            'total_dist_qz': np.mean(np.array(total_dist_qz)),
            'mean_test_rec': np.zeros(A),
            'predicted_label': predicted_label,
            'data_indx': data_indx,
            'z_prob': z_prob,
            'z_sample': z_sample,
            'x_low': data_low,
            'recon_c': recon_cell,
            'prune_indx': prune_indx
        }


    def save_file(self, fname, **kwargs):
        """
        Save data as a .p file using pickle.

        input args
            fname: the path of the pre-trained network.
            kwarg: keyword arguments for input variables e.g., x=[], y=[], etc.
        """

        f = open(fname + '.p', "wb")
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

        data = pickle.load(open(fname + '.p', "rb"))
        return data


# 50 epochs
