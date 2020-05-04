import os
import torch
from algo import train_vae
from vae import VQVAE

cfd = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cfd, 'data')
result_dir = os.path.join(cfd, 'results')
model_dir = os.path.join(result_dir, 'models')

hp_ckpt = torch.load(os.path.join(model_dir, 'hp.pt'))
archive_d_filename = os.path.join(model_dir, 'archive_d.pt')
archive_d = torch.load(archive_d_filename)
vqvae_filename = os.path.join(model_dir, 'vqvae.pt')
vqvae_ckpt = torch.load(vqvae_filename)
vqvae = VQVAE(
    hp_ckpt["dde_elite"]["nb_links"],
    hp_ckpt["vqvae"]["K"],
    hp_ckpt["vqvae"]["D"],
    hp_ckpt["vqvae"]["nb_hidden"],
    hp_ckpt["vqvae"]["nb_dim_hidden"],
    hp_ckpt["vqvae"]["nb_z"],
    hp_ckpt["vqvae"]["use_rezero"],
)

vqvae = train_vae(
    1000,
    archive_d,
    hp_ckpt["vqvae"]["K"],
    hp_ckpt["vqvae"]["D"],
    hp_ckpt["vqvae"]["nb_z"],
    hp_ckpt["vqvae"]["nb_hidden"],
    hp_ckpt["vqvae"]["nb_dim_hidden"],
    use_rezero=hp_ckpt["vqvae"]["use_rezero"],
    print_message=True,
    vqvae=vqvae
)

torch.save(vqvae.state_dict(), os.path.join(model_dir, 'retrained_vqvae.pt'))
