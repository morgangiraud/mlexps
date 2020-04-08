"""
Experiement where we explore the paper:
Automating Representation Discovery with MAP-Elites

Link: https://arxiv.org/pdf/2003.04389.pdf

We replicate the paper with one change, instead of a VAE, we use a
VQVAE.
"""

import os
# import neptune
import numpy as np
import torch
from algo import train_vae, map_elite
from functions import (
    behaviour_func,
    fitness_func,
    create_iso_mut,
    create_line_mut,
    create_recon_mut
)
from archive import Archive

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

# DDE ELITE HP
# nb_links = 20
nb_links = 100
nb_dde_elite_iter = 100
nb_bins = 30**2
iso_mut_sigma = 0.2
line_mut_sigma1 = 0.1
line_mut_sigma2 = 0.2

# MAP ELITE HP
nb_iter = 10  # The number of children will be nb_iter * nb_samples
nb_samples = 50

# VAE HP
nb_vae_epochs = 30
nb_final_vae_epochs = 200
K = 32
D = 3
nb_z = 3
nb_hidden = 5
nb_dim_hidden = 512
use_rezero = True  # Great for stability and for increasing the lr

cfd = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cfd, 'data')
model_dir = os.path.join(cfd, 'models')
result_dir = os.path.join(cfd, 'results', 'nb_links_{}'.format(nb_links))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

archive = Archive(nb_links, nb_bins)
archive.init(behaviour_func, fitness_func)

create_mut_operators_dict = {
    'create_iso_mut': create_iso_mut,
    'create_line_mut': create_line_mut,
    'create_recon_mut': create_recon_mut,
}

successes = np.ones(len(create_mut_operators_dict))
selections = np.ones(len(create_mut_operators_dict))

for i in range(nb_dde_elite_iter):
    print(
        'Generation {} -> successes: {}, selections: {}'.format(
            i, successes, selections
        )
    )

    # UCB algorithm
    probs = successes / selections + np.sqrt(
        2 * np.log(np.sum(successes)) / selections
    )

    var_idx = np.argmax(probs)
    mut_key = list(create_mut_operators_dict.keys())[var_idx]
    create_mut_op = create_mut_operators_dict[mut_key]

    if mut_key == 'create_recon_mut':
        # We train a VAE only if we are going to use it in that iteration
        vae = train_vae(
            nb_vae_epochs,
            archive.get_elites(),
            K,
            D,
            nb_z,
            nb_hidden,
            nb_dim_hidden,
            use_rezero,
            False,
        )
        mut_op = create_mut_op(vae)  # type: ignore
    elif mut_key == 'create_iso_mut':
        mut_op = create_mut_op(iso_mut_sigma)  # type: ignore
    elif mut_key == 'create_line_mut':
        mut_op = create_mut_op(line_mut_sigma1, line_mut_sigma2)  # type: ignore
    else:
        raise Exception('mut operator {} unknown'.format(mut_key))

    archive, s = map_elite(
        behaviour_func, fitness_func, mut_op, archive, nb_iter, nb_samples
    )

    successes[var_idx] += s
    selections[var_idx] += nb_iter * nb_samples

    archive.draw_illuminated_map(
        os.path.join(cfd, result_dir, '{0:02d}_illuminated_map.png'.format(i))
    )

# We train the final VAE
vqvae = train_vae(
    nb_final_vae_epochs,
    archive.get_elites(),
    K,
    D,
    nb_z,
    nb_hidden,
    nb_dim_hidden,
    use_rezero,
    True,
)

# Savings
torch.save(
    {
        'dde_elite': {
            'nb_links': nb_links,
            'nb_dde_elite_iter': nb_dde_elite_iter,
            'nb_bins': nb_bins,
            'iso_mut_sigma': iso_mut_sigma,
            'line_mut_sigma1': line_mut_sigma1,
            'line_mut_sigma2': line_mut_sigma2,
        },
        'map_elite': {
            'nb_iter': nb_iter,
            'nb_samples': nb_samples, },
        'vqvae': {
            'nb_vae_epochs': nb_vae_epochs,
            'nb_final_vae_epochs': nb_final_vae_epochs,
            'K': K,
            'D': D,
            'nb_z': nb_z,
            'nb_hidden': nb_hidden,
            'nb_dim_hidden': nb_dim_hidden,
        }
    },
    os.path.join(model_dir, 'hp.pt')  # yapf: ignore
)
torch.save(vqvae.state_dict(), os.path.join(model_dir, 'vqvae.pt'))
torch.save(archive.d, os.path.join(model_dir, 'archive_d.pt'))
