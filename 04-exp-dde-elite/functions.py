import torch
import numpy as np
from archive import Archive
from vae import VQVAE
from typing import Tuple, Callable


def create_iso_mut(sigma: float) -> Callable:
    def iso_mut(N: int, archive: Archive) -> Tuple:
        """
        isotropique mutation.

        Add isotropic noise to the elite parameters
        """
        x = archive.sample(N)
        mut = np.random.standard_normal(x.shape)

        return (x, None), x + sigma * mut

    return iso_mut


def create_line_mut(sigma1: float, sigma2: float) -> Callable:
    def line_mut(N: int, archive: Archive) -> Tuple:
        x = archive.sample(N)
        y = archive.sample(N)

        mut1 = np.random.standard_normal(x.shape)
        mut2 = np.random.standard_normal(x.shape)

        return (x, y), x + sigma1 * mut1 + sigma2 * (y - x) * mut2

    return line_mut


def create_recon_mut(vae: VQVAE) -> Callable:
    def recon_mut(N: int, archive: Archive) -> Tuple:
        x = archive.sample(N)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = vae.decode(vae.encode(x_tensor))
        y = y_tensor.numpy()

        return (x, y), (x + y) / 2

    return recon_mut


def behaviour_func(x: np.ndarray, links: np.ndarray) -> np.ndarray:
    assert links.shape[0] == x.shape[1]
    b = np.zeros([x.shape[0], 2])
    for i in range(x.shape[1]):
        c1 = np.cos(links[i] * np.sum(x[:, :i + 1], axis=1))
        c2 = np.sin(links[i] * np.sum(x[:, :i + 1], axis=1))
        b[:, 0] = b[:, 0] + c1
        b[:, 1] = b[:, 1] + c2

    b = b.round(4)

    return b


def fitness_func(x: np.ndarray) -> np.ndarray:
    means = np.mean(x, axis=1, keepdims=True)
    neg_var = -np.mean((x - means)**2, axis=1)

    return neg_var


def perplexity(sample_indices, nb_classes):
    r"""
    Compute the perplexity of sample using a
    uniform distribution
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = sample_indices.view(-1, 1)
    one_hot = torch.zeros([x.shape[0], nb_classes]).to(device)
    one_hot = one_hot.scatter(1, x, 1)

    probs = torch.mean(one_hot, axis=0)
    epsilon = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + epsilon))

    return torch.exp(entropy)
