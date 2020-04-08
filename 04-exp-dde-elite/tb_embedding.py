import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from vae import VQVAE
# import matplotlib
# from matplotlib import pyplot as plot
# matplotlib.use("Agg")

cfd = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cfd, 'data')
model_dir = os.path.join(cfd, 'models')
result_dir = os.path.join(cfd, 'results')

hp_ckpt = torch.load(os.path.join(model_dir, 'hp.pt'))
vqvae_filename = os.path.join(model_dir, 'vqvae.pt')
vqvae_ckpt = torch.load(vqvae_filename)
vqvae = VQVAE(
    hp_ckpt["dde_elite"]["nb_links"],
    hp_ckpt["vqvae"]["K"],
    hp_ckpt["vqvae"]["D"],
    hp_ckpt["vqvae"]["nb_hidden"],
    hp_ckpt["vqvae"]["nb_dim_hidden"],
    hp_ckpt["vqvae"]["nb_z"]
)
vqvae.load_state_dict(vqvae_ckpt)
e = vqvae.quant._embedding

writer = SummaryWriter(log_dir=result_dir)
labels = np.array(list(range(e.weight.shape[0])))
writer.add_embedding(e.weight)
writer.close()

# U, V = zip(*e.weight.detach().numpy())
# X = np.zeros(len(U))
# Y = np.zeros(len(U))
# plot.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
# ax = plot.gca()
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# plot.draw()
# plot.show()
