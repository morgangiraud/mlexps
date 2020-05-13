"""
Experiment on a vector quantized VAE

We explore VQ-VAE on MNIST using the reconstruction loss and
auto-regressive decoder.
The prior is first, a uniform distribution and then a fitted
auto-regressive model.
"""
import os
import neptune
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VQVAE
from functions import perplexity, compute_nb_vector_used
import torchvision

cfd = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cfd, 'data')
model_dir = os.path.join(cfd, 'models')
result_dir = os.path.join(cfd, 'results')

torch.manual_seed(1)

train_set = datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (1.0, ))])
)
test_set = datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (1.0, ))])
)

# Training HP
batch_size = 256
nb_vqvae_epochs = 10
nb_prior_epochs = 20
# num_training_updates = 15000
learning_rate = 1e-3

# Model HP
nb_res_hchannels = 32

# Embedding HP
K = 512
D = 64

# Prior HP
rnn_hidden_dim = 256
rnn_num_layers = 2

# Loss HP
commitment_cost = 0.25
# decay = 0.99

training_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
validation_loader = DataLoader(test_set, batch_size=32, shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae = VQVAE(K, D, nb_res_hchannels, rnn_hidden_dim, rnn_num_layers).to(device)
optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)

neptune.init('morgangiraud/sandbox')
hp = {
    'batch_size': batch_size,
    'nb_vqvae_epochs': nb_vqvae_epochs,
    'nb_prior_epochs': nb_prior_epochs,
    'learning_rate': learning_rate,
    'nb_res_hchannels': nb_res_hchannels,
    'K': K,
    'D': D,
    'commitment_cost': commitment_cost,
    'rnn_hidden_dim': 256,
    'rnn_num_layers': 2,
}
with neptune.create_experiment(
    name='03-exp-vq-vae-mnist', description='Learning a Vq VAE and fitting its prior', params=hp
):
    # Train the VQ-VAE
    for epoch in range(nb_vqvae_epochs):
        print("\n=====")
        print("VQ VAE training, epoch: {}".format(epoch))

        for i_batch, (x, _) in enumerate(training_loader):
            x = x.to(device)
            optimizer.zero_grad()

            z_e, z_q_st, z_q, x_hat, indices = vqvae(x)
            assert x_hat.shape == x.shape

            recon_loss = F.mse_loss(x, x_hat)
            commit_loss = F.mse_loss(z_e, z_q.detach())
            embed_loss = F.mse_loss(z_e.detach(), z_q)
            loss = recon_loss + embed_loss + commitment_cost * commit_loss

            loss.backward()
            optimizer.step()

            if i_batch % 5 == 0:
                p = perplexity(indices.to(device), K)
                neptune.send_metric('Loss/loss', loss.item())
                neptune.send_metric('Loss/recon_loss', recon_loss.item())
                neptune.send_metric('Loss/embed_loss', commit_loss.item())
                neptune.send_metric('Perplexity', p.item())
                if i_batch % 20 == 0:
                    print(
                        """Loss: {}, recon_loss: {}, \
                            embed_loss: {}, perplexity: {}""".format(
                            loss.item(), recon_loss.item(), embed_loss.item(), p.item()
                        )
                    )

    torch.save(hp, os.path.join(model_dir, 'hp.pt'))
    torch.save(vqvae.state_dict(), os.path.join(model_dir, 'vqvae.pt'))

    nb_vector_used = compute_nb_vector_used(vqvae, training_loader)
    print("nb_vector_used: {}".format(nb_vector_used))

    _, _, _, x_hat, _ = vqvae(x)
    x_img_grid = torchvision.utils.make_grid(x, batch_size // 4)
    x_hat_img_grid = torchvision.utils.make_grid(x_hat, batch_size // 4)
    torchvision.utils.save_image(x_img_grid, os.path.join(cfd, result_dir, 'x_img_gen.png'))
    torchvision.utils.save_image(x_hat_img_grid, os.path.join(cfd, result_dir, 'x_hat_img_gen.png'))

    # Fit the prior
    for epoch in range(nb_prior_epochs):
        print("\n=====")
        print("Prior training, epoch: {}".format(epoch))

        for i_batch, (x, _) in enumerate(training_loader):
            x = x.to(device)

            optimizer.zero_grad()

            code = vqvae.encode(x).detach()

            one_hot = torch.zeros(code.shape[0], code.shape[1], K).to(device)
            one_hot = one_hot.scatter(2, code.unsqueeze(-1), 1)  # bs x H_l*W_l x K
            first_input = torch.zeros_like(one_hot[:, 1, :]).unsqueeze(1).to(device)
            lstm_inputs = torch.cat([first_input, one_hot[:, :-1, :]], 1)

            logits, _ = vqvae._p_z(lstm_inputs)

            loss = F.cross_entropy(logits.view(-1, K), code.view(-1))

            loss.backward()
            optimizer.step()

            if i_batch % 5 == 0:
                neptune.send_metric('Loss/prior_loss', loss.item())
                if i_batch % 20 == 0:
                    print("Loss: {}".format(loss.item()))

    # Create a sample
    imgs, latent_code = vqvae.sample(batch_size)

    img_grid = torchvision.utils.make_grid(imgs, batch_size // 4)
    torchvision.utils.save_image(img_grid, os.path.join(cfd, result_dir, 'img_gen.png'))
