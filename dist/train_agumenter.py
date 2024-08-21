import argparse
import os
from mmidas.augmentation.train import *
from mmidas.utils.tools import get_paths
from mmidas.augmentation.networks import Augmenter, Discriminator
from mmidas.utils.dataloader import load_data, get_loaders


# Setup argument parser for command line arguments
parser = argparse.ArgumentParser()

# Define command line arguments
parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
parser.add_argument("--batch_size", default=5000, type=int, help="batch size")
parser.add_argument("--lr", default=.001, type=float, help="learning rate")
parser.add_argument("--alpha", default=0.2, type=float, help="triplet loss hyperparameter")
parser.add_argument("--ws", default=[1, 0.5, 0.1, 0.5], type=list, help="weights of the augmenter loss")
parser.add_argument("--dim_noise", default=50, type=int, help="noise dimension")
parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
parser.add_argument("--loss_mode", default='MSE', type=str, help="loss mode, MSE or ZINB")
parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
parser.add_argument("--device", default=None, type=int, help="computing device, either 'cpu' or 'cuda'.")



# Main function
def main(latent_dim, n_epoch, lr, alpha, ws, dim_noise, batch_size, loss_mode, device):

    # Load configuration paths
    toml_file = 'pyproject.toml'
    sub_file = 'smartseq_files'
    config = get_paths(toml_file=toml_file, sub_file=sub_file)
    data_path = config['paths']['main_dir'] / config['paths']['data_path']
    data_file = data_path / config[sub_file]['anndata_file']

    # Define folder name for saving results
    folder_name = f'run_1'
    
    saving_folder = config['paths']['main_dir'] / config['paths']['saving_path']
    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)

    if loss_mode == 'MSE':
        n_zim = 1
    else:
        n_zim = 2

    # Dictionary of the training parameters for CTX-HIP datatset
    parameters = {'batch_size': batch_size,  # batch size
                'num_epochs': n_epoch,  # number of epochs
                'learning_rate': lr, # learning rate
                'alpha': alpha,  # triplet loss hyperparameter
                'num_z': latent_dim, # latent space dimension
                'num_n': dim_noise, # noise dimension
                'lambda': ws, # weights of the augmenter loss
                'dataset_file': data_file,
                'mode': loss_mode,
                'initial_w': False, # initial weights
                'affine': False,
                'saving_path': saving_folder
                }
    
    # Load data
    data = load_data(datafile=data_file)
    trainloader, testloader, _ = get_loaders(dataset=data['log1p'], batch_size=batch_size)

    parameters['n_features'] = data.data['log1p'].shape[-1]

    netA = Augmenter(noise_dim=parameters['num_n'],
                    latent_dim=parameters['num_z'],
                    mode=parameters['mode'],
                    input_dim=parameters['n_features']).to(device)
    
    netD = Discriminator(input_dim=parameters['n_features']).to(device)

    train_udagan(parameters, trainloader, device)
