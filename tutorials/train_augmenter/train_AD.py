from utils.augmentation.train import *

# Use GPU if available
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

path = os.path.abspath(os.path.join(os.path.join(os.path.join(os.getcwd(), '..'), '..'), '..'))
# data_file = '/home/yeganeh/Remote-AI/MTG_AD_data/all_donors_data/AD_MTG_L2-3-IT_nGene_9881_nDonor_84.p'
n_gene = 11983 # 7688 # 9881
subclass = 'Pvalb'
data_file = f'/allen/programs/celltypes/workgroups/mousecelltypes/Yeganeh/MTG_AD_data/all_donors_data/AD_MTG_{subclass}_nGene_{n_gene}_nDonor_84.p'
saving_path = '/allen/programs/celltypes/workgroups/mousecelltypes/Yeganeh/'
# os.makedirs(saving_path + '/results/augmenter', exist_ok=True)
# saving_path = saving_path + '/results/augmenter/'

# Dictionary of the training parameters for CTX-HIP datatset
parameters = {'batch_size': 1000,  # batch size
            'num_epochs': 1000,  # number of epochs
            'learning_rate': 1e-3, # learning rate
            'alpha': 0.2,  # triplet loss hyperparameter
            'num_z': 10, # latent space dimension
            'num_n': 50, # noise dimension
            'lambda': [1, 0.5, 0.1, 0.5], # weights of the augmenter loss
            'dataset_file': data_file,
            'feature': 'log1p',
            'subclass': '',
            'n_zim': 2,
            'n_smp': 20, # number of augmented samples
            'initial_w': False, # initial weights
            'affine': False,
            'n_run': 1,
            'save': 'True', # saving flag
            'file_name':  saving_path + 'model_Pvalb_zdim_2_D_10_ngene_' + str(n_gene),
            'saving_path': saving_path
            }

train_udagan(parameters, device)
