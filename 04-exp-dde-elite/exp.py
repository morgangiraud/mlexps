"""
Experiement where we explore the paper:
Automating Representation Discovery with MAP-Elites

Link: https://arxiv.org/pdf/2003.04389.pdf

We replicate the paper with one change, instead of a VAE, we use a
VQVAE.
"""

import os
import shutil
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
from bandit_window import BanditWindow

should_continue = input(
    "This script will delete the results folder, should we continue (y/[n])?"
)
if not (should_continue == 'y'):
    exit(0)

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

# DDE ELITE HP
nb_links = 20
# nb_links = 100
nb_dde_elite_gen = 300
nb_bins = 45**2  # ~ 1950 reacheable point
iso_mut_sigma = 0.003
line_mut_sigma = 0.1

# MAP ELITE HP
nb_iter = 1  # The number of children will be nb_iter * nb_samples
nb_samples = 100
bandit_window_size = 100

# VAE HP
nb_vae_epochs = 5
nb_final_vae_epochs = 250
K = 32
D = 3
nb_z = 3
nb_hidden = 5
nb_dim_hidden = 512
use_rezero = True  # Great for stability and for increasing the lr

cfd = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cfd, 'data')
result_dir = os.path.join(cfd, 'results')
model_dir = os.path.join(result_dir, 'models')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
shutil.rmtree(result_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

archive = Archive(nb_links, nb_bins)
archive.init(behaviour_func, fitness_func)

create_mut_operators_dict = {
    'create_iso_mut': create_iso_mut,
    'create_line_mut': create_line_mut,
    'create_recon_mut': create_recon_mut,
}
# This is equivalent to 3 bandit choices: [1, 0, 0], [0, 1, 0], [0, 0, 1]
nb_choice = len(create_mut_operators_dict)
bandit = BanditWindow(nb_choice, bandit_window_size)

for i in range(nb_dde_elite_gen):
    print('Generation {} -> {}'.format(i, bandit))

    # UCB algorithm
    logits = bandit.get_logits()
    var_idx = np.argmax(logits)

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
            use_rezero=use_rezero,
            print_message=False,
        )
        mut_op = create_mut_op(vae)  # type: ignore
    elif mut_key == 'create_iso_mut':
        mut_op = create_mut_op(iso_mut_sigma)  # type: ignore
    elif mut_key == 'create_line_mut':
        mut_op = create_mut_op(line_mut_sigma, iso_mut_sigma)  # type: ignore
    else:
        raise Exception('mut operator {} unknown'.format(mut_key))

    archive, s = map_elite(
        behaviour_func, fitness_func, mut_op, archive, nb_iter, nb_samples
    )

    s_avg = s / (nb_iter * nb_samples)
    bandit.add_result(s_avg, var_idx)

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
    use_rezero=use_rezero,
    print_message=True,
)

# Savings
torch.save(
    {
        'dde_elite': {
            'nb_links': nb_links,
            'nb_dde_elite_gen': nb_dde_elite_gen,
            'nb_bins': nb_bins,
            'iso_mut_sigma': iso_mut_sigma,
            'line_mut_sigma': line_mut_sigma,
        },
        'map_elite': {
            'nb_iter': nb_iter,
            'nb_samples': nb_samples,
            'bandit_window_size': bandit_window_size,
        },
        'vqvae': {
            'nb_vae_epochs': nb_vae_epochs,
            'nb_final_vae_epochs': nb_final_vae_epochs,
            'K': K,
            'D': D,
            'nb_z': nb_z,
            'nb_hidden': nb_hidden,
            'nb_dim_hidden': nb_dim_hidden,
            'use_rezero': use_rezero,
        }
    },
    os.path.join(model_dir, 'hp.pt')  # yapf: ignore
)
torch.save(vqvae.state_dict(), os.path.join(model_dir, 'vqvae.pt'))
torch.save(archive.d, os.path.join(model_dir, 'archive_d.pt'))
