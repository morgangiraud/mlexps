import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class VAE(nn.Module):
    r"""Variationnal autoencoder."""
    def __init__(self):
        super(VAE, self).__init__()

        self.nb_latent = 8
        self.L_mask = torch.tensor([[0, 0], [1, 0]])

        input_dim = 100
        h_dims = [2**6, 2**5, 2**4]
        layers = []
        for h_dim in h_dims:
            layers.append(nn.Sequential(nn.Linear(input_dim, h_dim), nn.ReLU()))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, self.nb_latent))
        self.encoder = torch.nn.Sequential(*layers)

        self.dist = MultivariateNormal(torch.zeros(2), torch.eye(2))

        self.fc1_up = nn.Linear(2, 2**5)
        self.fc2_up = nn.Linear(2**5, 2**6)
        self.fc3_up = nn.Linear(2**6, 100)

    def encode(self, x):
        r"""
        Encode the data into the parameters of the
        latent distribution.

        Args:
            x (Tensor): Input tensor (BS x D)

        Return:
            (List[Tensor]): list of parameters
        """
        x = self.encoder(x)

        mus = x[:, :2]  # bs x 2
        diag_logvars = x[:, 2:4]  # bs x 2
        L_prime = x[:, 4:].view(-1, 2, 2)  # bs x 2 x 2
        L = self.L_mask * L_prime + torch.diag_embed(
            torch.exp(0.5 * diag_logvars)
        )  # bs x 2 x 2

        return [mus, diag_logvars, L]

    def reparam(self, mus, L):
        r"""
        Use the reparametrization trick to sample
        The latent distribution

        Args:
            mus (Tensor): means of the latent distributions
            L (Tensor): Covariance Cholesky decomposition of the latent
                distributions

        Return:
            (Tensor): Original noise
            (Tensor): Latent samples
        """

        noise = self.dist.sample([mus.shape[0]])  # bs x 2
        z = mus + torch.matmul(L, noise.unsqueeze(-1)).squeeze()  # bs x 2

        return noise, z

    def decode(self, z):
        """
        Decode the latent samples into the parameters
        of the distribution over data

        Args:
            z (Tensor): Latent samples

        Return:
            (Tensor): parameters
        """
        y = F.relu(self.fc1_up(z))
        y = F.relu(self.fc2_up(y))
        p = torch.sigmoid(self.fc3_up(y))

        return p

    def forward(self, x):
        mus, diag_logvars, L = self.encode(x)
        noise, z = self.reparam(mus, L)
        p = self.decode(z)

        return (mus, diag_logvars, L, noise, z, p)
