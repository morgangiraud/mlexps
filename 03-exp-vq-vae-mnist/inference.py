import os
import torch
import torchvision
from model import VQVAE

cfd = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(cfd, 'models')
result_dir = os.path.join(cfd, 'results')

hp_ckpt = torch.load(os.path.join(model_dir, 'hp.pt'))
vqvae_ckpt = torch.load(os.path.join(model_dir, 'vqvae.pt'))

torch.manual_seed(1)

vqvae = VQVAE(
    hp_ckpt["K"],
    hp_ckpt["D"],
    hp_ckpt["nb_res_hchannels"],
    hp_ckpt["rnn_hidden_dim"],
    hp_ckpt["rnn_num_layers"]
)
vqvae.load_state_dict(vqvae_ckpt)
bs = 32

imgs, latent_code = vqvae.sample(bs)

img_grid = torchvision.utils.make_grid(imgs, bs // 4)
torchvision.utils.save_image(img_grid, os.path.join(result_dir, 'img_gen.png'))
