"""
Utilitary.
"""

import os
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Categorical
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use("Agg")

cfd = os.path.dirname(os.path.realpath(__file__))
results_dir = os.path.join(cfd, 'results')


def jcfd(filename):
    """
    Join with the current file directory.

    Returns:
        path (string): the file full path
    """

    return os.path.join(results_dir, filename)


def rand_covariance(size):
    if type(size) != np.ndarray:
        size = np.array(size)
    assert len(size.shape) == 1
    assert size.shape[0] == 2
    assert size[0] == size[1]

    A = np.random.standard_normal(size)
    cov = torch.tensor(np.dot(A.T, A), dtype=torch.float32)

    return cov


def save_data_picture(data, filename):
    df = pd.DataFrame(data, columns=["x1", "x2", "cluster_index"])
    plot = sns.scatterplot(x="x1", y="x2", hue="cluster_index", data=df)
    image_path = jcfd(filename)
    plot.get_figure().savefig(image_path)
    plot.get_figure().clf()

    return image_path


def generate_dataset(binary=False):
    """
    Generate a dataset using a mixture of 3 2D gaussians
    This data is then projected in a 100 dimensional space
    Finally we apply the cosine non-linearity.
    Optionnaly, one can also make teh data binary as a last transformation

    Args:
        binary (bool): To binarise the data or not

    Return:
        dataset (torch.tensor): the dataset
    """
    ps = [
        MultivariateNormal(torch.tensor([1., 1.]), rand_covariance([2, 2])),
        MultivariateNormal(torch.tensor([-1., 0.]), rand_covariance([2, 2])),
        MultivariateNormal(torch.tensor([1., -2.]), rand_covariance([2, 2])),
    ]
    cat = Categorical(torch.rand([3]))
    X = np.array([
        np.concatenate([ps[c].sample([1]).numpy()[0], [c]])
        for c in cat.sample([1000])
    ])

    # Non linear projections into a 100d space
    proj = 3 * np.random.rand(2, 100)
    dataset = np.tanh(np.dot(X[:, :2], proj))

    if binary is True:
        dataset[dataset > 0] = 1
        dataset[dataset <= 0] = 0

    return torch.tensor(dataset, dtype=torch.float32), X
