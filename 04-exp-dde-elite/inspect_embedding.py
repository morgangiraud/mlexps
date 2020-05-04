import os
from itertools import combinations_with_replacement as cwr
import torch
import seaborn as sns
from vae import VQVAE
from functions import behaviour_func, fitness_func
from archive import Archive
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")

cfd = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(cfd, 'data')
result_dir = os.path.join(cfd, 'results')
model_dir = os.path.join(result_dir, 'models')


def compute_cos_sim(X):
    X_norm = torch.norm(X, dim=1, keepdim=True)  # N_e x 1
    X_norm_2 = torch.matmul(X_norm, X_norm.T)
    X_cos_sim = torch.matmul(X, X.T) / X_norm_2

    return X_cos_sim


hp_ckpt = torch.load(os.path.join(model_dir, 'hp.pt'))
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
vqvae.load_state_dict(vqvae_ckpt)
e_weight = vqvae.quant._embedding.weight  # N_e x d_e
N_e = e_weight.shape[0]

e_cos = compute_cos_sim(e_weight)
ax = sns.heatmap(e_cos.detach().numpy(), vmin=-1, vmax=1)
ax.get_figure().savefig(os.path.join(model_dir, 'embedding_cosine_simi.png'))
plt.clf()

# Reorder
e_weight = e_weight[e_cos[0, :].argsort(descending=True)]
e_cos = compute_cos_sim(e_weight)

group_size = []
e_order = torch.tensor([])
threshold = 0.95
while e_cos.shape[0] > 0:
    # Group
    sim_vec = e_cos[0] > threshold
    group_size.append(int(torch.sum(sim_vec).detach().numpy()))
    # import pdb;pdb.set_trace()
    e_order = torch.cat([e_order, e_weight[sim_vec]], dim=0)

    # Remove chosen vectors
    to_keep = e_cos[0] <= threshold
    e_cos = e_cos[to_keep]
    e_cos = e_cos.T[to_keep]  # e_cos is symmetric
    e_weight = e_weight[to_keep]

e_order_vec_norm = torch.norm(e_order, dim=1, keepdim=True)  # N_e x 1
d_order_cos = torch.matmul(e_order, e_order.T) / torch.matmul(
    e_order_vec_norm, e_order_vec_norm.T
)

print(
    """
Number of groups: {}
Group sizes: {}
""".format(len(group_size), group_size)
)
ax = sns.heatmap(d_order_cos.detach().numpy(), vmin=-1, vmax=1)
ax.get_figure().savefig(
    os.path.join(model_dir, 'embedding_cosine_simi_ordered.png')
)
plt.clf()

# Check dimension 1
a1 = Archive(hp_ckpt["dde_elite"]["nb_links"], hp_ckpt["dde_elite"]["nb_bins"])
gens = torch.tensor([])
for i in range(N_e):
    i_1 = e_order[i]
    i_2 = e_order[group_size[0]]
    i_3 = e_order[group_size[0] + group_size[1]]
    gen = torch.cat([i_1, i_2, i_3], dim=0).reshape(1, -1)

    gens = torch.cat([gens, gen], dim=0)
elites = vqvae._decoder(gens).detach().numpy()  # N_e x d_elite
b, coords = behaviour_func(elites, a1.links)
f = fitness_func(elites)
a1.update(elites, a1.compute_pos(b), f)
a1.draw_illuminated_map(os.path.join(model_dir, 'e_dim_{}.png'.format(0)))

cmap = plt.cm.get_cmap('hsv', N_e)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
for i in range(N_e):
    for j in range(coords.shape[1]):
        plt.plot(coords[i, j, 0], coords[i, j, 1], 'o', c=cmap(i))
plt.savefig(os.path.join(model_dir, 'e_dim_{}_arm.png'.format(0)))
plt.clf()

# Check dimension 2
a1 = Archive(hp_ckpt["dde_elite"]["nb_links"], hp_ckpt["dde_elite"]["nb_bins"])
gens = torch.tensor([])
for i in range(N_e):
    i_1 = e_order[group_size[0]]
    i_2 = e_order[i]
    i_3 = e_order[group_size[0] + group_size[1]]
    gen = torch.cat([i_1, i_2, i_3], dim=0).reshape(1, -1)

    gens = torch.cat([gens, gen], dim=0)
elites = vqvae._decoder(gens).detach().numpy()  # N_e x d_elite
b, coords = behaviour_func(elites, a1.links)
f = fitness_func(elites)
a1.update(elites, a1.compute_pos(b), f)
a1.draw_illuminated_map(os.path.join(model_dir, 'e_dim_{}.png'.format(1)))

cmap = plt.cm.get_cmap('hsv', N_e)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
for i in range(N_e):
    for j in range(coords.shape[1]):
        plt.plot(coords[i, j, 0], coords[i, j, 1], 'o', c=cmap(i))
plt.savefig(os.path.join(model_dir, 'e_dim_{}_arm.png'.format(1)))
plt.clf()

# Check dimension 3
a1 = Archive(hp_ckpt["dde_elite"]["nb_links"], hp_ckpt["dde_elite"]["nb_bins"])
gens = torch.tensor([])
for i in range(N_e):
    i_1 = e_order[group_size[0]]
    i_2 = e_order[group_size[0] + group_size[1]]
    i_3 = e_order[i]
    gen = torch.cat([i_1, i_2, i_3], dim=0).reshape(1, -1)

    gens = torch.cat([gens, gen], dim=0)
elites = vqvae._decoder(gens).detach().numpy()  # N_e x d_elite
b, coords = behaviour_func(elites, a1.links)
f = fitness_func(elites)
a1.update(elites, a1.compute_pos(b), f)
a1.draw_illuminated_map(os.path.join(model_dir, 'e_dim_{}.png'.format(2)))

cmap = plt.cm.get_cmap('hsv', N_e)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
for i in range(N_e):
    for j in range(coords.shape[1]):
        plt.plot(coords[i, j, 0], coords[i, j, 1], 'o', c=cmap(i))
plt.savefig(os.path.join(model_dir, 'e_dim_{}_arm.png'.format(2)))
plt.clf()

a1 = Archive(hp_ckpt["dde_elite"]["nb_links"], hp_ckpt["dde_elite"]["nb_bins"])
cmap = plt.cm.get_cmap('hsv', group_size[0]**3)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
for i, c in enumerate(cwr(range(group_size[1]), 3)):
    i_1 = e_order[group_size[0] + c[0]]
    i_2 = e_order[group_size[0] + c[1]]
    i_3 = e_order[group_size[0] + c[2]]
    gen = torch.cat([i_1, i_2, i_3], dim=0).reshape(1, -1)
    elite = vqvae._decoder(gen).detach().numpy()  # bs x d_elite
    b, coords = behaviour_func(elite, a1.links)
    f = fitness_func(elite)
    a1.update(elite, a1.compute_pos(b), f)

    for j in range(coords.shape[1]):
        plt.plot(coords[0, j, 0], coords[0, j, 1], 'o', c=cmap(i))
plt.savefig(os.path.join(model_dir, 'group_{}_arm_combinations.png'.format(1)))
plt.clf()

a1 = Archive(hp_ckpt["dde_elite"]["nb_links"], hp_ckpt["dde_elite"]["nb_bins"])
cmap = plt.cm.get_cmap('hsv', group_size[2]**3)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
for i, c in enumerate(cwr(range(group_size[2]), 3)):
    i_1 = e_order[group_size[0] + group_size[1] + c[0]]
    i_2 = e_order[group_size[0] + group_size[1] + c[1]]
    i_3 = e_order[group_size[0] + group_size[1] + c[2]]
    gen = torch.cat([i_1, i_2, i_3], dim=0).reshape(1, -1)
    elite = vqvae._decoder(gen).detach().numpy()  # bs x d_elite
    b, coords = behaviour_func(elite, a1.links)
    f = fitness_func(elite)
    a1.update(elite, a1.compute_pos(b), f)

    for j in range(coords.shape[1]):
        plt.plot(coords[0, j, 0], coords[0, j, 1], 'o', c=cmap(i))
plt.savefig(os.path.join(model_dir, 'group_{}_arm_combinations.png'.format(2)))
plt.clf()

a1 = Archive(hp_ckpt["dde_elite"]["nb_links"], hp_ckpt["dde_elite"]["nb_bins"])
cmap = plt.cm.get_cmap('hsv', group_size[3]**3)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
for i, c in enumerate(cwr(range(group_size[3]), 3)):
    i_1 = e_order[group_size[0] + group_size[1] + group_size[2] + c[0]]
    i_2 = e_order[group_size[0] + group_size[1] + group_size[2] + c[1]]
    i_3 = e_order[group_size[0] + group_size[1] + group_size[2] + c[2]]
    gen = torch.cat([i_1, i_2, i_3], dim=0).reshape(1, -1)
    elite = vqvae._decoder(gen).detach().numpy()  # bs x d_elite
    b, coords = behaviour_func(elite, a1.links)
    f = fitness_func(elite)
    a1.update(elite, a1.compute_pos(b), f)

    for j in range(coords.shape[1]):
        plt.plot(coords[0, j, 0], coords[0, j, 1], 'o', c=cmap(i))
plt.savefig(os.path.join(model_dir, 'group_{}_arm_combinations.png'.format(3)))
plt.clf()
