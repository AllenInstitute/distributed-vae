import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mmidas.augmentation.aug_utils import *

class Augmenter(nn.Module):
    def __init__(self, noise_dim, latent_dim, input_dim, n_dim=500, mode='MSE', p_drop=0.5):
        super().__init__()

        moment = 0.01
        self.dp = nn.Dropout(p_drop)
        self.mode = mode
        self.noise_dim = noise_dim

        self.noise = nn.Linear(noise_dim, noise_dim, bias=False)
        self.bnz = nn.BatchNorm1d(self.noise.out_features)

        self.fc1 = nn.Linear(input_dim, input_dim // 5)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc3 = nn.Linear(self.fc1.out_features, n_dim)
        self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc4 = nn.Linear(n_dim, n_dim)
        self.batch_fc4 = nn.BatchNorm1d(num_features=self.fc4.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc5 = nn.Linear(n_dim, n_dim // 5)
        self.batch_fc5 = nn.BatchNorm1d(num_features=self.fc5.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc5n = nn.Linear(n_dim + noise_dim, n_dim // 5)
        self.batch_fc5n = nn.BatchNorm1d(num_features=self.fc5n.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc_mu = nn.Linear(n_dim // 5, latent_dim)
        self.fc_sigma = nn.Linear(n_dim // 5, latent_dim)
        self.batch_fc_mu = nn.BatchNorm1d(num_features=self.fc_mu.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc6 = nn.Linear(self.fc_mu.out_features, n_dim // 5)
        self.batch_fc6 = nn.BatchNorm1d(num_features=self.fc6.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc7 = nn.Linear(self.fc6.out_features, n_dim)
        self.batch_fc7 = nn.BatchNorm1d(num_features=self.fc7.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc8 = nn.Linear(n_dim, n_dim)
        self.batch_fc8 = nn.BatchNorm1d(num_features=self.fc8.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc9 = nn.Linear(n_dim, input_dim // 5)
        self.batch_fc9 = nn.BatchNorm1d(num_features=self.fc9.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc10 = nn.Linear(input_dim // 5, input_dim // 5)
        self.batch_fc10 = nn.BatchNorm1d(num_features=self.fc10.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc11 = nn.Linear(self.fc10.out_features, input_dim)
        if self.mode == 'ZINB':
            self.fc11_p = nn.Linear(self.fc10.out_features, input_dim)

    def forward(self, x, batched, noise, scale=1.):
        if batched:
            x = F.relu(self.batch_fc1(self.fc1(self.dp(x)).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc2(self.fc2(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc3(self.fc3(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc4(self.fc4(x).permute(1, 2, 0))).permute(2, 0, 1)
            if noise:
                z = scale * th.randn(x.size(0), x.size(1), self.noise_dim, device=x.device)
                z = F.elu(self.bnz(self.noise(z).permute(1, 2, 0))).permute(2, 0, 1)
                x = th.cat((x, z), dim=2)
                x = F.relu(self.batch_fc5n(self.fc5n(x).permute(1, 2, 0))).permute(2, 0, 1)
            else:
                x = F.relu(self.batch_fc5(self.fc5(x).permute(1, 2, 0))).permute(2, 0, 1)
            mu = self.batch_fc_mu(self.fc_mu(x).permute(1, 2, 0)).permute(2, 0, 1)
            sigma = th.sigmoid(self.fc_sigma(x))
            s = reparam_trick(mu, sigma, mu.device)
            x = F.relu(self.batch_fc6(self.fc6(s).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc7(self.fc7(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc8(self.fc8(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc9(self.fc9(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc10(self.fc10(x).permute(1, 2, 0))).permute(2, 0, 1)
            if self.mode == 'ZINB':
                x_mu = F.relu(self.fc11(x))
                x_p = th.sigmoid(self.fc11_p(x))
                return s, th.cat((x_mu, x_p), dim=2)
            else:
                return s, F.relu(self.fc11(x))

        else:
            x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
            x = F.relu(self.batch_fc2(self.fc2(x)))
            x = F.relu(self.batch_fc3(self.fc3(x)))
            x = F.relu(self.batch_fc4(self.fc4(x)))
            if noise:
                z = scale * th.randn(x.size(0), self.noise_dim, device=device)
                z = F.elu(self.bnz(self.noise(z)))
                x = th.cat((x, z), dim=1)
                x = F.relu(self.batch_fc5n(self.fc5n(x)))
            else:
                x = F.relu(self.batch_fc5(self.fc5(x)))

            mu = self.batch_fc_mu(self.fc_mu(x))
            sigma = th.sigmoid(self.fc_sigma(x))
            s = reparam_trick(mu, sigma, mu.device)
            x = F.relu(self.batch_fc6(self.fc6(s)))
            x = F.relu(self.batch_fc7(self.fc7(x)))
            x = F.relu(self.batch_fc8(self.fc8(x)))
            x = F.relu(self.batch_fc9(self.fc9(x)))
            x = F.relu(self.batch_fc10(self.fc10(x)))
            if self.mode  == 'ZINB':
                x_mu = F.relu(self.fc11(x))
                x_p = th.sigmoid(self.fc11_p(x))
                return s, th.cat((x_mu, x_p), dim=1)
            else:
                return s, F.relu(self.fc11(x))

class Discriminator(nn.Module):
    def __init__(self, input_dim, p_drop=0.2):
        super().__init__()

        moment = 0.01
        self.dp = nn.Dropout(p_drop)

        self.fc1 = nn.Linear(input_dim, input_dim // 5)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features, eps=1e-10, momentum=moment, affine=False)
        # self.fc3 = nn.Linear(n_dim, n_dim)
        # self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features, eps=1e-10, momentum=moment, affine=False)
        self.disc = nn.Linear(self.fc2.out_features, 1, 1)

    def forward(self, x):

        x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
        x = F.relu(self.batch_fc2(self.fc2(x)))
        #x = F.relu(self.batch_fc3(self.fc3(x)))
        output = th.sigmoid(self.disc(x))
        return x, output


class Generator(nn.Module):
    def __init__(self, latent_dim, input_dim, n_dim=100, n_zim=1, p_drop=0.1):
        super().__init__()

        moment = 0.01
        self.dp = nn.Dropout(p_drop)
        self.n_zim = n_zim

        self.fc1 = nn.Linear(input_dim, n_dim)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc2 = nn.Linear(n_dim, n_dim)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc3 = nn.Linear(n_dim, n_dim)
        self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc_mu = nn.Linear(n_dim, latent_dim)
        self.fc_sigma = nn.Linear(n_dim, latent_dim)
        self.batch_fc_mu = nn.BatchNorm1d(num_features=self.fc_mu.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc6 = nn.Linear(self.fc_mu.out_features, n_dim)
        self.batch_fc6 = nn.BatchNorm1d(num_features=self.fc6.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc7 = nn.Linear(self.fc6.out_features, n_dim)
        self.batch_fc7 = nn.BatchNorm1d(num_features=self.fc7.out_features, eps=1e-10, momentum=moment, affine=False)
        # self.fc8 = nn.Linear(n_dim, n_dim)
        # self.batch_fc8 = nn.BatchNorm1d(num_features=self.fc8.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc10 = nn.Linear(n_dim, n_dim)
        self.batch_fc10 = nn.BatchNorm1d(num_features=self.fc10.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc11 = nn.Linear(self.fc10.out_features, input_dim)
        if self.n_zim > 1:
            self.fc11_p = nn.Linear(self.fc10.out_features, input_dim)

    def forward(self, x, device):
        x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
        x = F.relu(self.batch_fc2(self.fc2(x)))
        x = F.relu(self.batch_fc3(self.fc3(x)))
        mu = self.batch_fc_mu(self.fc_mu(x))
        sigma = th.sigmoid(self.fc_sigma(x))
        s = reparam_trick(mu, sigma, device)
        x = F.relu(self.batch_fc6(self.fc6(s)))
        x = F.relu(self.batch_fc7(self.fc7(x)))
        x = F.relu(self.batch_fc10(self.fc10(x)))
        if self.n_zim > 1:
            x_mu = F.relu(self.fc11(x))
            x_p = th.sigmoid(self.fc11_p(x))
            # x_eps = self.sigmoid(self.fc11_eps[arm](h10))
            return s, th.cat((x_mu, x_p), dim=1)
        else:
            return s, F.relu(self.fc11(x))
        # return s, F.relu(self.fc11(x))


class Augmenter_smartseq(nn.Module):
    def __init__(self, noise_dim, latent_dim, input_dim=5000, n_dim=500, p_drop=0.5):
        super().__init__()

        moment = 0.01
        self.noise_dim = noise_dim

        self.dp = nn.Dropout(p_drop)

        self.noise = nn.Linear(noise_dim, noise_dim, bias=False)
        self.bnz = nn.BatchNorm1d(self.noise.out_features)

        self.fc1 = nn.Linear(input_dim, input_dim // 5)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features, eps=1e-10, momentum=moment, affine=False)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc3 = nn.Linear(self.fc2.out_features, n_dim)
        self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc4 = nn.Linear(n_dim, n_dim)
        self.batch_fc4 = nn.BatchNorm1d(num_features=self.fc4.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc5 = nn.Linear(n_dim + noise_dim, n_dim // 5)
        self.batch_fc5 = nn.BatchNorm1d(num_features=self.fc5.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc_mu = nn.Linear(self.fc5.out_features, latent_dim)
        self.fc_sigma = nn.Linear(self.fc5.out_features, latent_dim)
        self.batch_fc_mu = nn.BatchNorm1d(num_features=self.fc_mu.out_features,
                                          eps=1e-10, momentum=moment,
                                          affine=False)
        self.fc6 = nn.Linear(self.fc_mu.out_features, n_dim // 5)
        self.batch_fc6 = nn.BatchNorm1d(num_features=self.fc6.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc7 = nn.Linear(self.fc6.out_features, n_dim)
        self.batch_fc7 = nn.BatchNorm1d(num_features=self.fc7.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc8 = nn.Linear(n_dim, n_dim)
        self.batch_fc8 = nn.BatchNorm1d(num_features=self.fc8.out_features,
                                        eps=1e-10, momentum=moment, affine=False)

        self.fc9 = nn.Linear(n_dim, input_dim // 5)
        self.batch_fc9 = nn.BatchNorm1d(num_features=self.fc9.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc10 = nn.Linear(self.fc9.out_features, self.fc9.out_features)
        self.batch_fc10 = nn.BatchNorm1d(num_features=self.fc10.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc11 = nn.Linear(self.fc10.out_features, input_dim)

    
    def forward(self, x, batched, scale=1., noise=True):
        if batched:
            z = th.randn(x.shape[0], x.shape[1], self.noise_dim, device=x.device, requires_grad=False)
            z = F.elu(self.bnz(self.noise(z).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc1(self.fc1(self.dp(x)).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc2(self.fc2(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc3(self.fc3(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc4(self.fc4(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = th.cat((x, z), dim=2)
            x = F.relu(self.batch_fc5(self.fc5(x).permute(1, 2, 0))).permute(2, 0, 1)
            mu = self.batch_fc_mu(self.fc_mu(x).permute(1, 2, 0)).permute(2, 0, 1)
            sigma = th.sigmoid(self.fc_sigma(x))
            s = reparam_trick(mu, sigma, mu.device)
            x = F.relu(self.batch_fc6(self.fc6(s).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc7(self.fc7(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc8(self.fc8(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc9(self.fc9(x).permute(1, 2, 0))).permute(2, 0, 1)
            x = F.relu(self.batch_fc10(self.fc10(x).permute(1, 2, 0))).permute(2, 0, 1)
        else:
            z = th.randn(x.shape[0], self.noise_dim, device=x.device, requires_grad=False)
            z = F.elu(self.bnz(self.noise(z)))
            x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
            x = F.relu(self.batch_fc2(self.fc2(x)))
            x = F.relu(self.batch_fc3(self.fc3(x)))
            x = F.relu(self.batch_fc4(self.fc4(x)))
            x = th.cat((x, z), dim=1)
            x = F.relu(self.batch_fc5(self.fc5(x)))
            mu = self.batch_fc_mu(self.fc_mu(x))
            sigma = th.sigmoid(self.fc_sigma(x))
            s = reparam_trick(mu, sigma, mu.device)
            x = F.relu(self.batch_fc6(self.fc6(s)))
            x = F.relu(self.batch_fc7(self.fc7(x)))
            x = F.relu(self.batch_fc8(self.fc8(x)))
            x = F.relu(self.batch_fc9(self.fc9(x)))
            x = F.relu(self.batch_fc10(self.fc10(x)))
        return s, F.relu(self.fc11(x))

