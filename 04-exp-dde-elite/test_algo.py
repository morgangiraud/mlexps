import unittest
import numpy as np
import torch

from algo import train_vae, map_elite
from archive import Archive
from functions import create_iso_mut, behaviour_func, fitness_func


class TestAlgo(unittest.TestCase):
    def test_train_vae(self):
        np.random.seed(1)
        torch.manual_seed(1)

        archive = Archive(10)
        elites = np.random.rand(100, archive.nb_links)
        archive.update(
            elites, np.arange(elites.shape[0]), -np.ones(elites.shape[0])
        )
        nb_epochs = 2
        K = 8
        D = 2
        nb_z = 2
        nb_hidden = 1
        nb_dim_hidden = 16

        _ = train_vae(
            nb_epochs,
            archive.get_elites(),
            K,
            D,
            nb_z,
            nb_hidden,
            nb_dim_hidden
        )

    def test_map_elite(self):
        np.random.seed(1)
        torch.manual_seed(1)

        archive = Archive(10)
        elites = np.random.rand(10, archive.nb_links)
        archive.update(
            elites, np.arange(elites.shape[0]), -np.ones(elites.shape[0])
        )

        sigma = 0.1
        nb_iter = 2
        nb_samples = 1

        archive, successes = map_elite(
            behaviour_func,
            fitness_func,
            create_iso_mut(sigma),
            archive,
            nb_iter,
            nb_samples
        )
        assert successes == 1
        assert archive.get_elites().shape[0] == 11
