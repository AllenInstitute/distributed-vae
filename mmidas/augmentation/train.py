import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time, os
from mmidas.augmentation.udagan import *
from mmidas.augmentation.dataloader import get_data
from mmidas.augmentation.aug_utils import *
# from module.model_sprites import *

eps = 1e-8

def train_udagan(data, parameters, device):

    dataloader = get_data(data=data, batch_size=parameters['batch_size'])
                             # datafile=parameters['dataset_file']) #  n_feature=parameters['n_features'], gene_id=parameters['feature'],  #key=parameters['feature'])

    parameters['n_features'] = dataloader.dataset.tensors[0].size(-1)
    netA = Augmenter(noise_dim=parameters['num_n'],
                         latent_dim=parameters['num_z'],
                        #  n_zim=parameters['n_zim'],
                         input_dim=parameters['n_features']).to(device)
    netD = Discriminator(input_dim=parameters['n_features']).to(device)

    iter_num = len(dataloader)

    if parameters['initial_w']:
        print('use initial weigths')
        netA.apply(weights_init)
        netD.apply(weights_init)

    # Loss functions
    criterionD = nn.BCELoss()
    mseDist = nn.MSELoss()

    # Set Adam optimiser for discriminator and augmenter
    optimD = optim.Adam([{'params': netD.parameters()}], lr=parameters['learning_rate'])
    optimA = optim.Adam([{'params': netA.parameters()}], lr=parameters['learning_rate'])

    real_label = 1.
    fake_label = 0.
    A_losses = []
    D_losses = []

    print('-'*50)
    print('Starting training ...')

    for epoch in range(parameters['num_epochs']):
        epoch_start_time = time.time()
        A_loss_e, D_loss_e = 0, 0
        gen_loss_e, recon_loss_e = 0, 0
        triplet_loss_e = 0
        n_adv = 0
        for i, (data, data_bin) in enumerate(dataloader, 0):
            b_size = parameters['batch_size']
            real_data = data.to(device)
            real_data_bin = data_bin.to(device)
            # Updating the discriminator -----------------------------------
            optimD.zero_grad()
            # Original samples
            label = torch.full((b_size,), real_label, device=device)
            _, probs_real = netD(real_data_bin)
            loss_real = criterionD(probs_real.view(-1), label)

            if F.relu(loss_real - np.log(2) / 2) > 0:
                loss_real.backward()
                optim_D = True
            else:
                optim_D = False

            # Augmented samples
            label.fill_(fake_label)
            noise = torch.randn(b_size, parameters['num_n'], device=device)
            # noise += 0.1 * torch.sign(noise)
            _, fake_data1 = netA(real_data, noise, True, device)
            # zeros = torch.zeros(b_size, parameters['num_n'], device=device)
            _, fake_data2 = netA(real_data, noise, False, device)
            # binarizing the augmented sample
            if parameters['n_zim'] > 1:
                # p_bern_1 = fake_data1[:, parameters['n_features']:]
                # p_bern_2 = fake_data2[:, parameters['n_features']:]
                p_bern_1 = real_data_bin * fake_data1[:, parameters['n_features']:]
                p_bern_2 = real_data_bin * fake_data2[:, parameters['n_features']:]
                # fake_data1_bin = real_data_bin * fake_data1
                # fake_data2_bin = real_data_bin * fake_data2
                    # p_bern_1 = fake_data1[:, parameters['n_features']:]
                    # p_bern_2 = fake_data2[:, parameters['n_features']:]
                fake_data1_bin = torch.bernoulli(p_bern_1)
                fake_data2_bin = torch.bernoulli(p_bern_2)
                fake_data = fake_data2[:, :parameters['n_features']] * real_data_bin
            else:
                fake_data1_bin = 0. * fake_data1
                fake_data2_bin = 0. * fake_data2
                fake_data1_bin[fake_data1 > 1e-3] = 1.
                fake_data2_bin[fake_data2 > 1e-3] = 1.
                fake_data = 1. * fake_data2

            _, probs_fake1 = netD(fake_data1_bin.detach())
            _, probs_fake2 = netD(fake_data2_bin.detach())
            loss_fake = (criterionD(probs_fake1.view(-1), label) + criterionD(probs_fake2.view(-1), label)) / 2

            if F.relu(loss_fake - np.log(2) / 2) > 0:
                loss_fake.backward()
                optim_D = True

            # Loss value for the discriminator
            D_loss = loss_real + loss_fake

            if optim_D:
                optimD.step()
            else:
                n_adv += 1

            # Updating the augmenter ---------------------------------------
            optimA.zero_grad()
            # Augmented data treated as real data
            z1, probs_fake1 = netD(fake_data1_bin)
            z2, probs_fake2 = netD(fake_data2_bin)
            # z0, _ = netD(real_data)
            label.fill_(real_label)
            gen_loss = (criterionD(probs_fake1.view(-1), label) + criterionD(probs_fake2.view(-1), label)) / 2
            triplet_loss = TripletLoss(real_data_bin.view(b_size, -1),
                                       fake_data2_bin.view(b_size, -1),
                                       fake_data1_bin.view(b_size, -1),
                                       parameters['alpha'], 'BCE')

            recon_loss = (F.mse_loss(fake_data, real_data, reduction='mean') + criterionD(fake_data2_bin, real_data_bin)) / 2
            # else:
            #     recon_loss = (F.mse_loss(fake_data2, real_data, reduction='mean') + criterionD(fake_data2_bin, real_data_bin)) / 2

            # Loss value for the augmenter
            A_loss = parameters['lambda'][0] * gen_loss + \
                     parameters['lambda'][1] * triplet_loss + \
                     parameters['lambda'][2] * mseDist(z1, z2) + \
                     parameters['lambda'][3] * recon_loss
            A_loss.backward()
            optimA.step()

            A_losses.append(A_loss.data.item())
            D_losses.append(D_loss.data.item())
            A_loss_e += A_loss.data.item()
            D_loss_e += D_loss.data.item()
            gen_loss_e += gen_loss.data.item()
            recon_loss_e += recon_loss.data.item()
            triplet_loss_e += triplet_loss.data.item()

        A_loss_epoch = A_loss_e / (iter_num)
        D_loss_epoch = D_loss_e / (iter_num )
        gen_loss_epoch = gen_loss_e / (iter_num)
        recon_loss_epoch = recon_loss_e / (iter_num)
        triplet_loss_epoch = triplet_loss_e / (iter_num)

        print('=====> Epoch:{}, Generator Loss: {:.4f}, Discriminator Loss: {'
              ':.4f}, Recon Loss: {:.4f}, Trip Loss: '
              '{:.4f}, Elapsed Time:{:.2f}'.format(epoch, A_loss_epoch,
                    D_loss_epoch, recon_loss_epoch, triplet_loss_epoch,
                    time.time() - epoch_start_time))

    print("-" * 50)
    # Save trained models
    if parameters['save']:

        torch.save({
            'netA': netA.state_dict(),
            'netD': netD.state_dict(),
            'optimD': optimD.state_dict(),
            'optimA': optimA.state_dict(),
            'parameters': parameters
            }, parameters['file_name'])

        # Plot the training losses.
        plt.figure()
        plt.title("Augmenter and Discriminator Loss Values in Training")
        plt.plot(A_losses, label="A")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(parameters['saving_path'] + 'loss_curve.png')
