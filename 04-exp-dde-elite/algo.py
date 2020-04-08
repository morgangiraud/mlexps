import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Tuple, Optional
from archive import Archive
from vae import VQVAE
from functions import perplexity


def graph_gradient_norm(
    model, writer: SummaryWriter, global_step: int, prefix: str = ''
):
    for name, p in model.named_parameters():
        # grad_norm = torch.norm(p.grad)
        # writer.add_scalar(
        #     name,
        #     grad_norm,
        #     global_step=global_step
        # )
        if re.search('alpha', name):
            writer.add_scalar(name, p, global_step=global_step)
            writer.add_scalar(
                '{}.grad'.format(name), p.grad, global_step=global_step
            )
        else:
            writer.add_histogram(name, p, global_step=global_step)
            writer.add_histogram(
                '{}.grad'.format(name), p.grad, global_step=global_step
            )


def train_vae(
    nb_epochs: int,
    elites: np.ndarray,
    K: int,
    D: int,
    nb_z: int,
    nb_hidden: int,
    nb_dim_hidden: int,
    use_rezero=False,
    print_message=False,
    vqvae: Optional[VQVAE] = None,
) -> VQVAE:
    batch_size = 32
    # learning_rate = 3e-3
    learning_rate = 1e-2

    # VQVAE param
    nb_links = elites.shape[1]

    commitment_cost = 0.25
    dataset = TensorDataset(torch.tensor(elites, dtype=torch.float32))
    training_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not vqvae:
        vqvae = VQVAE(
            nb_links,
            K,
            D,
            nb_hidden,
            nb_dim_hidden,
            nb_z,
            use_rezero=use_rezero
        ).to(device)
    else:
        if print_message:
            print('Using existing VQVAE')
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=0.3
    )

    if print_message:
        cfd = os.path.dirname(os.path.realpath(__file__))
        result_dir = os.path.join(
            cfd, 'results', 'nb_links_{},lr_{}'.format(nb_links, learning_rate)
        )
        writer = SummaryWriter(log_dir=result_dir)

    global_step = 0
    for epoch in range(nb_epochs):
        if print_message:
            print("\n=====")
            print("VQ VAE training, epoch: {}".format(epoch))
            current_lr = scheduler.get_lr()[0]  # type: ignore
            writer.add_scalar('lr', current_lr, global_step=epoch)
            writer.add_embedding(
                vqvae.quant._embedding.weight,
                global_step=epoch,
                tag='vqvae embedding'
            )

        for i_batch, (x, ) in enumerate(training_loader):
            global_step += 1
            x = x.to(device)

            z_e, z_q_st, z_q, x_hat, indices = vqvae(x)
            assert x_hat.shape == x.shape

            recon_loss = F.mse_loss(x, x_hat)
            commit_loss = F.mse_loss(z_e, z_q.detach())
            embed_loss = F.mse_loss(z_e.detach(), z_q)
            loss = recon_loss + embed_loss + commitment_cost * commit_loss

            optimizer.zero_grad()
            loss.backward()
            if print_message:
                if i_batch % 5 == 0:
                    graph_gradient_norm(vqvae, writer, global_step)
                    p = perplexity(indices.to(device), K)
                    if i_batch % 20 == 0:
                        print(
                            """
global step: {} -> Loss: {}, recon_loss: {}, embed_loss: {}, perplexity: {}
                            """.format(
                                global_step,
                                loss.item(),
                                recon_loss.item(),
                                embed_loss.item(),
                                p.item()
                            )
                        )
            optimizer.step()

        scheduler.step()  # type: ignore

    if print_message:
        writer.add_embedding(
            vqvae.quant._embedding.weight,
            global_step=global_step,
            tag='vqvae embedding'
        )
        writer.close()

    return vqvae


def map_elite(
    behaviour_f: Callable,
    fitness_f: Callable,
    mut_op: Callable,
    archive: Archive,
    nb_iter: int,
    nb_samples: int
) -> Tuple[Archive, int]:
    """
    MAP-elites alogirthm.

    I've added the capacity to control sequential and parallel
    iterations. Those are not the same. If you look at the algorithm
    in the paper, it seems full sequential but my guess is that it is slow.
    So why not do a little bit of both ?
    """
    successes = 0
    for i in range(nb_iter):
        (parent_x, parent_y), children = mut_op(nb_samples, archive)

        bs = behaviour_f(children, archive.links)
        ps = archive.compute_pos(bs)

        fs = fitness_f(children)

        successes += archive.update(children, ps, fs)

    return archive, successes
