import argparse
import os
import numpy as np
from utils.training import train_cplmixVAE
from utils.helpers import load_config, load_patchseq
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_categories", default=100, type=int, help="number of cell types")
parser.add_argument("--state_dim_T", default=2, type=int, help="state variable dimension for T data")
parser.add_argument("--state_dim_E", default=3, type=int, help="state variable dimension for E data")
parser.add_argument("--n_arm_T", default=2, type=int,  help="number of mixVAE arms for T modality")
parser.add_argument("--n_arm_E", default=2, type=int,  help="number of mixVAE arms for E modality")
parser.add_argument("--n_modal", default=2, type=int,  help="number of data modalities")
parser.add_argument("--temp",  default=1.0, type=float, help="gumbel-softmax temperature")
parser.add_argument("--tau",  default=.01, type=float, help="softmax temperature")
parser.add_argument("--beta",  default=1, type=float, help="beta factor")
parser.add_argument("--lam_T", default=1, type=int,  help="coupling factor between T arms")
parser.add_argument("--lam_E", default=1, type=int,  help="coupling factor between E arms")
parser.add_argument("--lam_TE", default=1, type=int,  help="coupling factor between TE arms")
parser.add_argument("--latent_dim_T", default=30, type=int, help="latent dimension of T")
parser.add_argument("--latent_dim_E", default=30, type=int, help="latent dimension of E")
parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
parser.add_argument("--n_epoch_p", default=10000, type=int, help="Number of epochs to train pruning algorithm")
parser.add_argument("--max_prun_it", default=60, type=int, help="maximum number of pruning iterations")
parser.add_argument("--min_con", default=.95, type=float, help="minimum consensus")
parser.add_argument("--min_density", default=5, type=int, help="minimum number of samples in a class")
parser.add_argument("--aug_file", default='augModel_T', type=str, help="path of the data augmenter")
parser.add_argument("--n_aug_smp", default=0, type=int, help="number of augmented samples")
parser.add_argument("--fc_dim_T", default=100, type=int, help="number of nodes at the hidden layers for T data")
parser.add_argument("--fc_dim_E", default=100, type=int, help="number of nodes at the hidden layers for E data")
parser.add_argument("--batch_size", default=1000, type=int, help="batch size")
parser.add_argument("--variational", default='true', type=str, help="enable variational mode")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--p_drop_T", default=0.25, type=float, help="input probability of dropout for T data")
parser.add_argument("--p_drop_E", default=0.0, type=float, help="input probability of dropout for E data")
parser.add_argument("--noise_std", default=0.05, type=float, help="additive noise std for E data")
parser.add_argument("--s_drop_T", default=0.0, type=float, help="state probability of dropout for T")
parser.add_argument("--s_drop_E", default=0.0, type=float, help="state probability of dropout for E")
parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
parser.add_argument("--hard", default='false', type=str, help="hard encoding")
parser.add_argument("--device", default=0, type=int, help="gpu device, use None for cpu")
parser.add_argument("--saving_folder", default='results', type=str, help="saving path")


def main(n_categories, n_arm_T, n_arm_E, n_modal, state_dim_T, state_dim_E, latent_dim_T, latent_dim_E, fc_dim_T, fc_dim_E, n_epoch, n_epoch_p, min_con, min_density,
         batch_size, p_drop_T, p_drop_E, s_drop_T, s_drop_E, noise_std, lr, temp, n_run, saving_folder, device, hard, tau, aug_file, variational, max_prun_it, beta,
         lam_T, lam_E, lam_TE, n_aug_smp):

    paths = load_config('config.toml')
    saving_folder = paths['package_dir'] / saving_folder
    folder_name = f'run_{n_run}_proc_K_{n_categories}_SdimT_{state_dim_T}_SdimE_{state_dim_E}_' + f'lr_{lr}_n_armT_{n_arm_T}_n_armE_{n_arm_E}_tau_{tau}_nbatch_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}'

    if aug_file:
        aug_file = 'augmenter/' + aug_file
        aug_file = saving_folder / aug_file

    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)

    state_dim = dict()
    state_dim['T'] = state_dim_T
    state_dim['E'] = state_dim_E
    fc_dim = dict()
    fc_dim['T'] = fc_dim_T
    fc_dim['E'] = fc_dim_E
    latent_dim = dict()
    latent_dim['T'] = latent_dim_T
    latent_dim['E'] = latent_dim_E
    p_drop = dict()
    p_drop['T'] = p_drop_T
    p_drop['E'] = p_drop_E
    s_drop = dict()
    s_drop['T'] = s_drop_T
    s_drop['E'] = s_drop_E
    n_arm = dict()
    n_arm['T'] = n_arm_T
    n_arm['E'] = n_arm_E
    lam = dict()
    lam['T'] = lam_T
    lam['E'] = lam_E
    lam['TE'] = lam_TE

    # load Patch-seq data
    rmv_cell = ['Sst Crh 4930553C11Rik           ',
                'Sst Myh8 Etv1                   ']
    D = load_patchseq(path=paths['data'], exclude_type=rmv_cell, min_num=5)
    data_T = D['XT']
    data_E = D['XE']
    mask = dict()
    # T_smp_id = D['sample_id_T']
    # E_smp_id = D['sample_id_E']
    # crossModal_id = D['crossModal_id']
    mask['ET'] = D['crossModal_id'] #[np.where(E_smp_id == c_id)[0][0] for c_id in crossModal_id]
    mask['TE'] = D['crossModal_id'] #[np.where(T_smp_id == c_id)[0][0] for c_id in crossModal_id]
    mask['E'] = ~D['isnan_E']
    mask['T'] = ~D['isnan_T']

    input_dim = dict()
    input_dim['T'] = data_T.shape[1]
    input_dim['E'] = data_E.shape[1]

    if variational == 'true':
        state_det = False
    else:
        state_det = True

    if hard == 'true':
        b_hard = True
    else:
        b_hard = False

    cpl_mixVAE = train_cplmixVAE(saving_folder=saving_folder, device=device, aug_file=aug_file)
    alldata_loader, train_loader, validation_loader, test_loader = cpl_mixVAE.getdata(dataset_T=data_T, dataset_E=data_E, label=D['cluster_label'], batch_size=batch_size, n_aug_smp=n_aug_smp)
    cpl_mixVAE.init_model(n_categories=n_categories,
                          state_dim=state_dim,
                          input_dim=input_dim,
                          fc_dim=fc_dim,
                          lowD_dim=latent_dim,
                          x_drop=p_drop,
                          s_drop=s_drop,
                          noise_std=noise_std,
                          state_det=state_det,
                          beta=beta,
                          lr=lr,
                          n_arm=n_arm,
                          n_modal=n_modal,
                          temp=temp,
                          hard=b_hard,
                          tau=tau,
                          lam=lam)

    cpl_mixVAE.run(train_loader, validation_loader, test_loader, alldata_loader, mask, n_epoch, n_epoch_p, min_con, max_prun_it, min_density)

    outcome_dict = cpl_mixVAE.eval_model(data_T, data_E, mask)
    plt.figure()
    tmp = np.concatenate(outcome_dict['state_mu']['T'][0])
    plt.scatter(tmp[:, 0], tmp[:, 1])
    plt.savefig(saving_folder + '/state_mu_T.png', dpi=600)
    plt.figure()
    tmp = np.concatenate(outcome_dict['state_mu']['E'][0])
    if state_dim_E > 1:
        plt.scatter(tmp[:, 0], tmp[:, 1])
        plt.savefig(saving_folder + '/state_mu_E.png', dpi=600)
    plt.close('all')

    # study role of the categorical variable in reconstruction
    # i = 105  # choose a cell
    # l_rec, l_rand_rec, recon, rand_recon, recon_change = cpl_mixVAE.cat_travers_analy(data_T[i, :], data_E[i, :])



if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
