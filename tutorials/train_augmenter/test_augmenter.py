import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.augmentation.udagan import *
from utils.augmentation.dataloader import get_data
from utils.augmentation.aug_utils import *
from matplotlib import gridspec
import seaborn as sns


fontsize = 6
num_gene = 9134
path = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(path)
data_file = path + '/MTG_AD_data/all_donors_data/AD_MTG_Sst_nGene_' + str(num_gene) + '_nDonor_84.p'
os.makedirs('./results/augmenter/' + 'genes_' + str(num_gene), exist_ok=True)
saving_path = './results/augmenter/genes_' + str(num_gene) + '/'
file_aug = './results/augmenter/model_Sst_zdim_2_D_10_ngene_' + str(num_gene)
load_file_aug = file_aug
# Load the checkpoint file
model_aug = torch.load(load_file_aug)

# Set the device to run on: GPU or CPU
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'parameters' dictionary from the loaded file
parameters = model_aug['parameters']
n_samp = parameters['n_smp']

dataloader, dataset = get_data(batch_size=parameters['batch_size'], file=data_file, key=parameters['feature'])

genes_id = np.arange(dataset[parameters['feature']].shape[1])
parameters['n_features'] = dataloader.dataset.tensors[0].size(-1)

# Initialise the networks
print('no affine functions ...')
netA = Augmenter(noise_dim=parameters['num_n'],
                    latent_dim=parameters['num_z'],
                    n_zim=parameters['n_zim'],
                    input_dim=parameters['n_features']).to(device)
netA.load_state_dict(model_aug['netA'])
n_a = 5
print(parameters['n_zim'])

for arm in range(2): #range(parameters['n_arm']):
    print('-----> sample:{}'.format(arm))

    data_real_low = np.zeros((len(dataloader.dataset), 2))
    data_aug_low = np.zeros((len(dataloader.dataset), 2))
    data_bin_low = np.zeros((len(dataloader.dataset), 2))
    gene_samp_real = np.zeros((len(dataloader.dataset), parameters['n_features']))
    gene_samp_aug = np.zeros((n_a, len(dataloader.dataset), parameters['n_features']))
    samp_clr = ["" for x in range(len(dataloader.dataset))]
    cluster = ["" for x in range(len(dataloader.dataset))]

    with torch.no_grad():
        for i, (data, data_bin) in enumerate(dataloader, 0):
            # Get batch size
            b_size = parameters['batch_size']
            # Generate augmented samples
            real_data = data.to(device)
            real_data_bin = data_bin.to(device)
            gene_samp_real[i * b_size:min((i + 1) * b_size, len(dataloader.dataset)), :] = real_data.cpu().detach().numpy()

            for smp in range(n_a):
                noise = torch.randn(b_size, parameters['num_n'], device=device)
                # noise += 0.1 * torch.sign(noise)
                _, gen_data = netA(real_data, noise, True, device)

                if parameters['n_zim'] > 1:
                    p_bern = gen_data[:, parameters['n_features']:]
                    # fake_data = gen_data[:, :parameters['n_features']] * torch.bernoulli(p_bern)
                    fake_data = gen_data[:, :parameters['n_features']] * real_data_bin
                else:
                    fake_data = gen_data.clone()

                gene_samp_aug[smp, i * b_size:min((i + 1) * b_size,  len(dataloader.dataset)), :] = fake_data.cpu().detach().numpy()

            fig, ax = plt.subplots(1, 2)
            im = ax[0].imshow(real_data[:, :100].detach().cpu().numpy(), aspect=.1)
            fig.colorbar(im, orientation='vertical')
            im = ax[1].imshow(fake_data[:, :100].cpu().detach().numpy(), aspect=.1)
            fig.colorbar(im, orientation='vertical')
            plt.savefig(saving_path + 'aug_vs_real_' + str(i) + '.png')


    samp_clr = samp_clr[:i*b_size]
    cluster = cluster[:i*b_size]
    data_real_low = data_real_low[:i*b_size, :]
    data_aug_low = data_aug_low[:i*b_size, :]
    data_bin_low = data_bin_low[:i * b_size, :]

# genes_id = dataset['gene_id'][gene_index]
plt.close('all')

for ii in range(500):
    df = pd.DataFrame()
    df['original'] = gene_samp_real[:, ii]
    df['augmented'] = gene_samp_aug[0, :, ii]
    fig = plt.figure(figsize=[12, 5])
    gs = gridspec.GridSpec(1, 1, hspace=0.1)
    ax0 = plt.subplot(gs[0])
    sns.histplot(df, bins=100, element='step', alpha=0.5)
    ax0.set_yscale('log')
    ax0.set_ylabel('Log of Counts')
    # xlim = ax0.get_xlim()
    # ax = plt.subplot(gs[1])
    # sns.histplot(gene_samp_real[:, ii], bins=100, element='step', ax=ax0)
    # ax.set_yscale('log')
    # ax.set_title(gg, fontsize=16)
    # ax.set_ylabel('Log of Counts')
    # ax = plt.subplot(gs[1])
    # sns.histplot(gene_samp_aug[0, :, ii], bins=100, element='step', ax=ax)
    # ax.set_yscale('log')
    # ax.set_ylabel('Log of Counts')
    # ax
    gs.tight_layout(fig)
    plt.savefig(saving_path + 'gene_' + str(ii) + '.png')
    plt.close()

