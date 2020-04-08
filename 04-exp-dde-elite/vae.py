import torch
import torch.nn as nn


class VectorQuantStraightThrough(nn.Module):
    def __init__(self, K, D):
        super(VectorQuantStraightThrough, self).__init__()

        self.K = K  # number of embedding vectors
        self.D = D  # dimension of embedding vectors
        self._embedding = nn.Embedding(self.K, self.D)
        self._embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, z_e):
        V = z_e.shape[1]
        C = z_e.shape[2]
        assert C == self.D

        pairwise_distances = torch.cdist(
            z_e.view(-1, C), self._embedding.weight
        )  # bs*V, 1
        indices = torch.argmin(pairwise_distances, dim=1)  # bs*V

        z_q = self._embedding(indices)  # bs*V x C
        z_q = z_q.view(z_e.shape)  # bs x V x C
        # To simulate a straight through gradient
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, z_q, indices.view(-1, V)


class ReZeroLinearBlock(nn.Module):
    def __init__(self, c):
        super(ReZeroLinearBlock, self).__init__()

        self.c = c
        self.alpha = torch.nn.Parameter(torch.Tensor([0.]))
        self.block = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(True), )

    def forward(self, x):
        out = x + self.alpha * self.block(x)

        return out


class VQVAE(nn.Module):
    r"""Vector Quantized Variationnal AutoEncoder"""
    def __init__(
        self, nb_links, K, D, nb_hidden, nb_dim_hidden, nb_z, use_rezero=False
    ):
        super(VQVAE, self).__init__()

        self.use_rezero = use_rezero

        # Encoder
        self.input_dim = (None, nb_links)
        self.z_e_size = (nb_z, D)
        self._encoder = self.build_encoder(
            self.input_dim, D, nb_hidden, nb_dim_hidden
        )

        # Quantizer
        self.quant = VectorQuantStraightThrough(K, D)

        # Decoder
        self._decoder = self.build_decoder(
            self.input_dim, D, nb_hidden, nb_dim_hidden
        )

        # Prior
        self._p_z = torch.distributions.Categorical(torch.ones(K) / K)

    def build_encoder(self, input_dim, D, nb_hidden, nb_dim_hidden):
        r"""Construct the encoder"""

        assert len(input_dim) == 2
        # Input_dim: bs x I

        previous_dim = input_dim[1]
        layers = []
        for i in range(nb_hidden):
            if previous_dim == nb_dim_hidden and self.use_rezero:
                layers.append(ReZeroLinearBlock(nb_dim_hidden))
            else:
                layers.append(nn.Linear(previous_dim, nb_dim_hidden))
                layers.append(nn.ReLU(True))
            previous_dim = nb_dim_hidden
        layers.append(
            nn.Linear(previous_dim, self.z_e_size[0] * self.z_e_size[1])
        )

        encoder = nn.Sequential(*layers)

        return encoder

    def build_decoder(self, input_dim, D, nb_hidden, nb_dim_hidden):
        r"""Construct the decoder"""

        previous_dim = self.z_e_size[0] * self.z_e_size[1]
        layers = []
        for i in range(nb_hidden):
            if previous_dim == nb_dim_hidden and self.use_rezero:
                layers.append(ReZeroLinearBlock(nb_dim_hidden))
            else:
                layers.append(nn.Linear(previous_dim, nb_dim_hidden))
                layers.append(nn.ReLU(True))
            previous_dim = nb_dim_hidden
        layers.append(nn.Linear(previous_dim, input_dim[1]))

        decoder = nn.Sequential(*layers)

        return decoder

    def forward(self, x):
        raw_z_e = self._encoder(x)
        z_e = torch.reshape(raw_z_e, (-1, ) + self.z_e_size)

        z_q_st, z_q, indices = self.quant(z_e)

        raw_z_q_st = torch.reshape(
            z_q_st, (-1, self.z_e_size[0] * self.z_e_size[1])
        )
        x_hat = self._decoder(raw_z_q_st)

        return z_e, z_q_st, z_q, x_hat, indices

    def encode(self, x):
        with torch.no_grad():
            raw_z_e = self._encoder(x)
            z_e = torch.reshape(raw_z_e, (-1, ) + self.z_e_size)

            _, _, latent_code = self.quant(z_e)

        return latent_code

    def decode(self, latent_code):
        with torch.no_grad():
            bs = latent_code.shape[0]

            z_q = self.quant._embedding(latent_code)

            raw_z_q = torch.reshape(
                z_q, (bs, self.z_e_size[0] * self.z_e_size[1])
            )
            x_hat = self._decoder(raw_z_q)

        return x_hat
