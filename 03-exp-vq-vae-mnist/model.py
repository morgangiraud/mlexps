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
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        W = z_e.shape[1]
        H = z_e.shape[2]
        C = z_e.shape[3]
        assert C == self.D

        pairwise_distances = torch.cdist(
            z_e.view(-1, C), self._embedding.weight
        )  # bs*H*W x K
        indices = torch.argmin(pairwise_distances, dim=1)  # bs*H*W x 1

        z_q = self._embedding(indices)  # bs*H*W x C
        z_q = z_q.view(z_e.shape)
        # To simulate a straight through gradient
        z_q_st = z_e + (z_q - z_e).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q_st = z_q_st.permute(0, 3, 1, 2).contiguous()

        return z_q_st, z_q, indices.view(-1, W * H)


class ResBlock(nn.Module):
    def __init__(self, in_channels, nb_res_hchannels):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        self.nb_res_hchannels = nb_res_hchannels
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                self.in_channels,
                self.nb_res_hchannels,
                3,
                padding=1,
                bias=False
            ),
            nn.ReLU(True),
            nn.Conv2d(self.nb_res_hchannels, self.in_channels, 1, bias=False),
        )

    def forward(self, x):
        out = x + self.block(x)

        return out


class AutoRegressive(nn.Module):
    def __init__(self, K, hidden_dim, num_layers=1):
        super(AutoRegressive, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_dim = hidden_dim
        self.K = K
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            self.K,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        hidden0 = torch.zeros(self.num_layers, 1, self.hidden_dim).to(device)
        h_init = torch.nn.Parameter(hidden0, requires_grad=True)
        c_init = torch.nn.Parameter(hidden0, requires_grad=True)
        self._init_state = (h_init, c_init)

        self.proj_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.K),
            nn.ReLU(True),
            nn.Linear(self.K, self.K),
        )

    def forward(self, seq, h_state=None):
        bs = seq.shape[0]
        if h_state is None:
            h_state = (
                self._init_state[0].repeat(1, bs, 1),
                self._init_state[1].repeat(1, bs, 1)
            )

        lstm_out, h_state = self.lstm(seq, h_state)
        logits = self.proj_out(
            lstm_out.reshape(bs * seq.shape[1], self.hidden_dim)
        )

        return logits.view(bs, seq.shape[1], self.K), h_state


class VQVAE(nn.Module):
    r"""Vector Quantized Variationnal AutoEncoder"""
    def __init__(self, K, D, nb_res_hchannels, rnn_hidden_dim, rnn_num_layers):
        super(VQVAE, self).__init__()

        # Encoder
        self.input_dim = (None, 1, 28, 28)
        self.z_e_size = (64, 7, 7)
        self._encoder = self.build_encoder(self.input_dim, nb_res_hchannels, D)

        # Quantizer
        self.quant = VectorQuantStraightThrough(K, D)

        # Decoder
        self._decoder = self.build_decoder(self.input_dim, nb_res_hchannels, D)

        # Prior
        self._p_z = AutoRegressive(K, rnn_hidden_dim, rnn_num_layers)

    def build_encoder(self, input_dim, nb_res_hchannels, D):
        r"""Construct the encoder"""

        assert len(input_dim) == 4
        # Input_dim: bs x C x H x W

        # MNIST case
        # bs x 1 x 28 x 28
        # bs x D/2 x 14 x 14
        # bs x D x 7 x 7

        encoder = nn.Sequential(
            nn.Conv2d(input_dim[1], D // 2, (4, 4), (2, 2), padding=1),
            nn.ReLU(True),
            nn.Conv2d(D // 2, D, (4, 4), (2, 2), padding=1),
            nn.ReLU(True),
            nn.Conv2d(D, D, (3, 3), (1, 1), padding=1),
            ResBlock(D, nb_res_hchannels),
            ResBlock(D, nb_res_hchannels),
            nn.ReLU(True)
        )

        return encoder

    def build_decoder(self, input_dim, nb_res_hchannels, D):
        r"""Construct the decoder"""

        decoder = nn.Sequential(
            nn.Conv2d(D, D, (3, 3), (1, 1), padding=1),
            ResBlock(D, nb_res_hchannels),
            ResBlock(D, nb_res_hchannels),
            nn.ReLU(True),
            nn.ConvTranspose2d(D, D // 2, (4, 4), (2, 2), padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(D // 2, input_dim[1], (4, 4), (2, 2), padding=1),
        )

        return decoder

    def forward(self, x):
        z_e = self._encoder(x)
        z_q_st, z_q, indices = self.quant(z_e)
        x_hat = self._decoder(z_q_st)

        return z_e, z_q_st, z_q, x_hat, indices

    def encode(self, x):
        with torch.no_grad():
            z_e = self._encoder(x)
            _, _, latent_code = self.quant(z_e)

        return latent_code

    def decode(self, latent_code):
        with torch.no_grad():
            bs = latent_code.shape[0]
            latent_code = latent_code.view(
                bs,
                self.z_e_size[1],
                self.z_e_size[2], ).long()

            z_q = self.quant._embedding(latent_code)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            x_hat = self._decoder(z_q)

        return x_hat

    def sample(self, bs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            x = torch.zeros(bs, 1, self.quant.K).to(device)
            h_state = None
            latent_code = None
            for i in range(self.z_e_size[1] * self.z_e_size[2]):
                logits, h_state = self._p_z(x, h_state)
                probs = torch.softmax(logits, 2)

                indice = torch.multinomial(probs.squeeze(), 1)
                x = torch.zeros(bs, 1, self.quant.K).to(device)
                x = x.scatter(2, indice.view(bs, 1, 1), 1)

                if latent_code is None:
                    latent_code = indice.unsqueeze(1)
                else:
                    latent_code = torch.cat([latent_code, indice.unsqueeze(1)],
                                            1)
            imgs = self.decode(latent_code)

        return imgs, latent_code
