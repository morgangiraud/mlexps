"""
Experiment on a binary variationnal autoencodeur.

We explore VAE with maximum likelihood objective
on a 10d Bernouilli distribution
"""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import neptune
import matplotlib
import utils
from model import VAE

matplotlib.use("Agg")

ap = argparse.ArgumentParser()
args = vars(ap.parse_args())


# Define losses
def elbo_loss(x, diag_logvars, noise, z, p):
    """
    The likelihood loss derived from the ELBO objective.
    """
    bs = x.shape[0]
    assert diag_logvars.shape == (bs, 2)
    assert noise.shape == (bs, 2)
    assert z.shape == (bs, 2)
    assert p.shape == x.shape
    epsilon = 1e-5

    bs_log_2pi = bs * torch.log(2 * torch.tensor(np.pi))

    log_q_z_x = -0.5 * (
        bs_log_2pi + torch.sum(diag_logvars, axis=1)
        + torch.matmul(noise.unsqueeze(1), noise.unsqueeze(-1)).squeeze()
    )
    assert log_q_z_x.shape == (bs, )

    # p_(z) is a prior distribution over z
    # In general we just use z ~ N(0, I)
    log_p_z = -0.5 * (
        bs_log_2pi + torch.matmul(z.unsqueeze(1), z.unsqueeze(-1))
    ).squeeze()
    if bs == 1:
        log_p_z = log_p_z.unsqueeze(0)
    assert log_p_z.shape == (bs, )

    log_p_theta_x_z = torch.sum(
        x * torch.log(p + epsilon) + (1 - x) * torch.log(1 - p + epsilon),
        axis=1
    )
    assert log_q_z_x.shape == (bs, )

    loss_p_z = -torch.mean(log_p_z)
    loss_p_theta_x_z = -torch.mean(log_p_theta_x_z)
    loss_q_z_x = -torch.mean(log_q_z_x)

    loss = loss_p_theta_x_z + loss_p_z - loss_q_z_x

    return loss, loss_p_z, loss_p_theta_x_z, loss_q_z_x


# Seed
torch.random.manual_seed(1)
np.random.seed(10)

# Generate the dataset using a mixtude of 3 2d gaussians
dataset, X = utils.generate_dataset(True)
bs = 32
dataloader = DataLoader(
    dataset,
    batch_size=bs,
    shuffle=True,
    num_workers=0, )
nb_epochs = 500

vae = VAE()
lr = 3e-2
milestones = [10, 100, 450]
lr_gamma = 0.1

neptune.init('morgangiraud/sandbox')
with neptune.create_experiment(
    name='02-exp-binary-vae',
    description='Experiment to check the VAE capacity to learn a multinormal',
    params={
        'lr': lr,
        'scheduler': 'MultiStepLR',
        'milestones': milestones,
        'gamma': lr_gamma,
        'batch_size': bs
    }
):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_gamma
    )
    # Lr -> 3e-2, 3e-3, 3e-4
    for epoch in range(nb_epochs):
        print("\n=====")
        print("epoch: {}, lr: {}".format(epoch, scheduler.get_lr()))
        for i_batch, sample_batched in enumerate(dataloader):
            optimizer.zero_grad()

            mus, diag_logvars, L, noise, z, p = vae.forward(sample_batched)
            loss, loss_p_z, loss_p_theta_x_z, loss_q_z_x = elbo_loss(
                sample_batched, diag_logvars, noise, z, p
            )

            loss.backward()
            optimizer.step()
            if i_batch % 20 == 0:
                print(
                    "Loss: {}, p_z: {}, p_theta_x_z: {}, q_z_x: {}".format(
                        loss.item(),
                        loss_p_z.item(),
                        loss_p_theta_x_z.item(),
                        loss_q_z_x.item(),
                    )
                )
            neptune.send_metric('Loss/loss', loss.item())
            neptune.send_metric('Loss/loss_p_z', loss_p_z.item())
            neptune.send_metric(
                'Loss/loss_p_theta_x_z', loss_p_theta_x_z.item()
            )
            neptune.send_metric('Loss/loss_q_z_x', loss_q_z_x.item())
            neptune.send_metric('lr', scheduler.get_lr()[0])
        scheduler.step()

    # Plot the dataset
    image_path = utils.save_data_picture(X, "dataset.png")
    neptune.send_artifact(image_path)

    # Let's plot all mapped Z
    full_dataloader = DataLoader(
        dataset,
        batch_size=dataset.shape[0],
        shuffle=False,
        num_workers=0, )
    for _, data in enumerate(full_dataloader):
        mus, _, L, _, Z, _ = vae.forward(data)
    Z = np.concatenate((Z.detach().numpy(), X[:, 2:]), 1)
    assert Z.shape == (dataset.shape[0], 3)

    image_path = utils.save_data_picture(Z, "mapped_z.png")
    neptune.send_artifact(image_path)

    # Let's plot the distributions arising from the
    # average of all the predicted distributions.
    mus_0 = mus[X[:, 2] == 0]
    count_0 = mus_0.shape[0]
    mu_0 = torch.mean(mus_0, 0, keepdim=True)
    L_0 = torch.mean(L[X[:, 2] == 0], 0, keepdim=True)
    noise_0 = vae.dist.sample([count_0])
    z_0 = mu_0 + torch.matmul(L_0, noise_0.unsqueeze(-1)).squeeze()  # bs x 2
    z_0 = torch.cat((z_0, torch.tensor([[0.]] * z_0.shape[0])), 1)

    mus_1 = mus[X[:, 2] == 1]
    count_1 = mus_1.shape[0]
    mu_1 = torch.mean(mus_1, 0, keepdim=True)
    L_1 = torch.mean(L[X[:, 2] == 0], 0, keepdim=True)
    noise_1 = vae.dist.sample([count_1])
    z_1 = mu_1 + torch.matmul(L_1, noise_1.unsqueeze(-1)).squeeze()  # bs x 2
    z_1 = torch.cat((z_1, torch.tensor([[1.]] * z_1.shape[0])), 1)

    mus_2 = mus[X[:, 2] == 2]
    count_2 = mus_2.shape[0]
    mu_2 = torch.mean(mus_2, 0, keepdim=True)
    L_2 = torch.mean(L[X[:, 2] == 0], 0, keepdim=True)
    noise_2 = vae.dist.sample([count_2])
    z_2 = mu_2 + torch.matmul(L_2, noise_2.unsqueeze(-1)).squeeze()  # bs x 2
    z_2 = torch.cat((z_2, torch.tensor([[2.]] * z_2.shape[0])), 1)

    Z_avg = torch.cat((z_0, z_1, z_2), 0)
    assert Z_avg.shape == (dataset.shape[0], 3)

    image_path = utils.save_data_picture(
        Z_avg.detach().numpy(), "average_z.png"
    )
    neptune.send_artifact(image_path)
