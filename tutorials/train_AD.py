import argparse
import os
from utils.training import train_cplmixVAE
import pickle
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

# AD_MTG_norm_L4-IT_ngene_7787.p
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='/data', type=str, help="data path") # MTG_AD_data/all_donors_data
parser.add_argument("--n_gene", default=9881, type=int, help="number of genes")
parser.add_argument("--n_categories", default=10, type=int, help="number of cell types")
parser.add_argument("--state_dim", default=10, type=int, help="state variable dimension")
parser.add_argument("--n_arm", default=2, type=int,  help="number of mixVAE arms for each modalities")
parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
parser.add_argument("--tau",  default=.1, type=float, help="softmax temperature")
parser.add_argument("--beta",  default=1, type=float, help="beta factor")
parser.add_argument("--latent_dim", default=30, type=int, help="latent dimension")
parser.add_argument("--n_epoch", default=1000, type=int, help="Number of epochs to train")
parser.add_argument("--n_epoch_p", default=1000, type=int, help="Number of epochs to train pruning algorithm")
parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
parser.add_argument("--max_pron_it", default=8, type=int, help="max number of pruning iteratoions")
parser.add_argument("--aug_path", default='./results/augmenter', type=str, help="path of the data augmenter")
parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=1000, type=int, help="batch size")
parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--p_drop", default=0.2, type=float, help="input probability of dropout")
parser.add_argument("--s_drop", default=0.0, type=float, help="state probability of dropout")
parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
parser.add_argument("--subclass", default='L2-3-IT', type=str, help="cell subclass, e.g. Sst")
parser.add_argument("--exclude_type", default=[''], type=str, help="L2/3 IT_2")
parser.add_argument("--exclude_donors", default=['UWA 7043', 'UWA 7157', 'UWA 6965'], type=str, help="['']")
parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
parser.add_argument("--device", default=0, type=int, help="gpu device, use None for cpu")
parser.add_argument("--saving_folder", default='./results/cpl_mixVAE/', type=str, help="saving path")

# ['UWA 7291', 'UWA 7245', 'UWA 6983', 'UWA 7163']
def main(data_path, n_gene, n_categories, n_arm, state_dim, latent_dim, fc_dim, n_epoch, n_epoch_p, min_con, max_pron_it, batch_size,
         p_drop, s_drop, lr, temp, n_run, saving_folder, device, hard, tau, aug_path, subclass, exclude_type, exclude_donors, variational, beta):

    path = os.getcwd() #os.path.abspath(os.path.join(os.getcwd(), '..'))
    # load data
    print(f'loading AD {subclass} data ... ')
    f = open(path + data_path + f'/AD_MTG_{subclass}_nGene_{n_gene}_nDonor_84.p', "rb")
    data = pickle.load(f)
    f.close()
    print('Data is loaded')

    if len(exclude_type) > 0:
        subclass_ind = np.array([i for i in range(len(data['supertype_scANVI'])) if data['supertype_scANVI'][i] not in exclude_type])
        print(np.unique(np.array(data['supertype_scANVI'])[subclass_ind]))
        ref_len = len(data['supertype_scANVI'])
        all_key = list(data.keys())
        for k in all_key:
            if len(data[k]) >= ref_len:
                if k == 'log1p':
                    data[k] = np.array(data[k])[subclass_ind, :]
                else:
                    data[k] = np.array(data[k])[subclass_ind]

    if len(exclude_donors) > 0:
        subcdata_ind = np.array([i for i in range(len(data['external_donor_name'])) if data['external_donor_name'][i] not in exclude_donors])
        print(np.unique(np.array(data['external_donor_name'])[subcdata_ind]))
        ref_len = len(data['supertype_scANVI'])
        all_key = list(data.keys())
        for k in all_key:
            if len(data[k]) >= ref_len:
                if k == 'log1p':
                    data[k] = np.array(data[k])[subcdata_ind, :]
                else:
                    data[k] = np.array(data[k])[subcdata_ind]

    print(data['log1p'].shape)

    folder_name = subclass + '_exc3Don_run_' + str(n_run) + '_K_' + str(n_categories) + '_Sdim_' + str(state_dim) + '_ngene_' + str(len(data['gene_id'])) + '_fcDim_' + \
                  str(fc_dim) + '_latDim_' + str(latent_dim) + '_lr_' + str(lr) + '_pDrop_' + str(p_drop) + '_n_arm_' + str(n_arm) + '_tau_' + str(tau) + \
                  '_bsize_' + str(batch_size) + '_nepoch_' + str(n_epoch) + '_nepochP_' + str(n_epoch_p)

    os.makedirs(saving_folder + folder_name, exist_ok=True)
    saving_folder = saving_folder + folder_name
    os.makedirs(saving_folder + '/model', exist_ok=True)

    aug_file = aug_path + f'/model_{subclass}_zdim_2_D_10_ngene_{n_gene}'
    cpl_mixVAE = train_cplmixVAE(saving_folder=saving_folder, device=device, aug_file=aug_file)
    alldata_loader, train_loader, val_set_torch, validation_loader, test_set_torch, test_loader = cpl_mixVAE.getdata(dataset=data['log1p'],
                                                                                                                     label=data['supertype_scANVI'], batch_size=batch_size)
    cpl_mixVAE.init_model(n_categories=n_categories, state_dim=state_dim, input_dim=data['log1p'].shape[1], fc_dim=fc_dim, lowD_dim=latent_dim, x_drop=p_drop,
                          s_drop=s_drop, lr=lr, n_arm=n_arm, temp=temp, hard=hard, tau=tau)
    cpl_mixVAE.run(train_loader, test_loader, val_set_torch, alldata_loader, n_epoch, n_epoch_p, min_con, max_pron_it)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
