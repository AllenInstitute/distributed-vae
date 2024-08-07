import argparse
import os
import numpy as np
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.tools import get_paths
from mmidas.utils.dataloader import load_data, get_loaders
from pathlib import Path

import wandb

def is_path(x):
    return isinstance(x, Path)

def wrap_in_path(x):
    return wrap_in_path(Path(x)) if not is_path(x) else x

# Main function
def main(n_categories, n_arm, state_dim, latent_dim, fc_dim, n_epoch, n_epoch_p, min_con, max_prun_it, batch_size, lam, lam_pc, loss_mode,
         p_drop, s_drop, lr, temp, n_run, device, hard, tau, variational, ref_pc, augmentation, pretrained_model, n_pr, beta, dataset):

    _args = locals()
    # try int(device):


    # Load configuration paths
    toml_file = 'pyproject.toml'
    config = get_paths(toml_file=toml_file, sub_file=dataset)
    data_file = wrap_in_path(config[dataset]['data_path']) / wrap_in_path(config[dataset]['anndata_file'])   

    # Define folder name for saving results
    folder_name = f'Run{n_run}_K{n_categories}_S{state_dim}_AUG{augmentation}_LR{lr}_A{n_arm}_B{batch_size}' + \
                  f'_E{n_epoch}_Ep{n_epoch_p}'
    saving_folder = config['paths']['main_dir'] / config[dataset]['saving_path'] / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)
    

    # Determine augmentation file path
    if augmentation:
        aug_file = config['paths']['main_dir'] / config[dataset]['aug_model']
    else:
        aug_file = ''

    # Determine pretrained model file path
    if pretrained_model:
        trained_model = config['paths']['main_dir'] / config[dataset]['trained_model']
    else:
        trained_model = ''

    # Load data
    data_dict = load_data(datafile=data_file)
    print("Data loaded successfully!")
    print(f"Number of cells: {data_dict['log1p'].shape[0]}, Number of genes: {data_dict['log1p'].shape[1]}")

    # Initialize the coupled mixVAE (MMIDAS) model
    cplMixVAE = cpl_mixVAE(saving_folder=saving_folder,
                                 device=device,
                                 aug_file=aug_file)

    # Make data loaders for training, validation, and testing
    fold = 0 # fold index for cross-validation, for reproducibility purpose
    alldata_loader, train_loader, validation_loader, test_loader = cplMixVAE.get_dataloader(dataset=data_dict['log1p'],
                                                                                             label=data_dict['cluster'],
                                                                                             batch_size=batch_size,
                                                                                             n_aug_smp=0,
                                                                                             fold=fold)

    # Initialize the model with specified parameters
    cplMixVAE.init_model(n_categories=n_categories,
                         state_dim=state_dim,
                         input_dim=data_dict['log1p'].shape[1],
                         fc_dim=fc_dim,
                         lowD_dim=latent_dim,
                         x_drop=p_drop,
                         s_drop=s_drop,
                         lr=lr,
                         n_arm=n_arm,
                         temp=temp,
                         hard=hard,
                         tau=tau,
                         lam=lam,
                         lam_pc=lam_pc,
                         beta=beta,
                         ref_prior=ref_pc,
                         variational=variational,
                         trained_model=trained_model,
                         n_pr=n_pr,
                         mode=loss_mode)

    # Train and save the model
    run = wandb.init(project='mmidas-arms', config=_args)
    model_file = cplMixVAE.train(train_loader=train_loader,
                                 test_loader=test_loader,
                                 n_epoch=n_epoch,
                                 n_epoch_p=n_epoch_p,
                                 c_onehot=data_dict['c_onehot'],
                                 c_p=data_dict['c_p'],
                                 min_con=min_con,
                                 max_prun_it=max_prun_it,
                                 run=run)


# Run the main function when the script is executed
if __name__ == "__main__":
    # Setup argument parser for command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_arm", default=2, type=int, help="number of mixVAE arms for each modality")

    parser.add_argument("--n_categories", default=115, type=int, help="number of cell types")
    parser.add_argument("--state_dim", default=2, type=int, help="state variable dimension")
    parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
    parser.add_argument("--tau",  default=.005, type=float, help="softmax temperature")
    parser.add_argument("--beta",  default=1, type=float, help="KL regularization parameter")
    parser.add_argument("--lam",  default=1, type=float, help="coupling factor")
    parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
    parser.add_argument("--n_epoch", default=20000, type=int, help="Number of epochs to train")
    parser.add_argument("--n_epoch_p", default=0, type=int, help="Number of epochs to train pruning algorithm")
    parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
    parser.add_argument("--max_prun_it", default=0, type=int, help="minimum number of samples in a class")
    parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
    parser.add_argument("--batch_size", default=5000, type=int, help="batch size")
    parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
    parser.add_argument("--augmentation", default=True, type=bool, help="enable VAE-GAN augmentation")
    parser.add_argument("--lr", default=.001, type=float, help="learning rate")
    parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
    parser.add_argument("--s_drop", default=0.0, type=float, help="state probability of dropout")
    parser.add_argument("--lam_pc", default=1, type=float, help="coupling factor for ref arm")
    parser.add_argument("--ref_pc", default=False, type=bool, help="use a reference prior component")
    parser.add_argument("--pretrained_model", default=False, type=bool, help="use pretrained model")
    parser.add_argument("--n_pr", default=0, type=int, help="number of pruned categories in case of using a pretrained model")
    parser.add_argument("--loss_mode", default='MSE', type=str, help="loss mode, MSE or ZINB")
    parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
    parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
    parser.add_argument("--dataset", default='mouse_smartseq', type=str, help="dataset name, e.g., 'mouse_smartseq', 'mouse_ctx_10x'")
    parser.add_argument("--device", default='cuda', type=str, help="computing device, either 'cpu' or 'cuda'.")

    args = parser.parse_args()
    main(**vars(args))
