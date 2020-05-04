import unittest

from functions import (
    create_iso_mut, create_line_mut, behaviour_func, fitness_func
)
from archive import Archive
import numpy as np


class TestIsoMut(unittest.TestCase):
    def test_iso_mut(self):
        np.random.seed(1)

        elites = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.], ], dtype=np.float)  # yapf: disable
        archive = Archive(2)
        archive.update(
            elites, np.arange(elites.shape[0]), -np.ones(elites.shape[0])
        )

        N = 3
        sigma = 0.1

        _, output = create_iso_mut(sigma)(N, archive)
        np.testing.assert_almost_equal(
            output, [[0.9472, 0.8927], [1.0865, -0.2302], [0.1745, -0.0761]],
            decimal=4
        )

        N = 10
        _, output = create_iso_mut(sigma)(N, archive)
        np.testing.assert_almost_equal(
            output,
            [[0.0197, 0.8455], [0.9912, 0.0852], [0.0677, -0.0107],
             [1.0725, 1.0935]],
            decimal=4
        )


class TestLineMut(unittest.TestCase):
    def test_line_mut(self):
        np.random.seed(1)

        elites = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.], ], dtype=np.float)  # yapf: disable
        archive = Archive(2)
        archive.update(
            elites, np.arange(elites.shape[0]), -np.ones(elites.shape[0])
        )

        N = 3
        sigma1 = 0.1
        sigma2 = 0.

        _, output = create_line_mut(sigma1, sigma2)(N, archive)
        np.testing.assert_almost_equal(
            output, [[0.8398, 0.8522], [1.0483, 0.0188], [-0.0465, 0.0862]],
            decimal=4
        )

        N = 10
        sigma1 = 0.
        sigma2 = 1.
        _, output = create_line_mut(sigma1, sigma2)(N, archive)
        np.testing.assert_almost_equal(
            output,
            [[1.8501, 0.0392], [-0.2174, 0.1585], [0.1266, -0.1114],
             [-1.038, 2.0095]],
            decimal=4
        )


class TestBehaviourFunc(unittest.TestCase):
    def test_behaviour_func(self):
        np.random.seed(1)
        x = np.array([
            [0., 0.],
            [0., np.pi / 2],
            [np.pi / 2, 0.],
            [-np.pi / 2., np.pi / 2],
        ], dtype=np.float)  # yapf: disable

        output, coords = behaviour_func(x, np.ones(x.shape[1]))

        np.testing.assert_almost_equal(
            output, [[2., 0.], [1., 1.], [0., 2.], [1., -1.]]
        )


class TestFitnessFunc(unittest.TestCase):
    def test_fitness_func(self):
        np.random.seed(1)
        x = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [-1., 1.], ], dtype=np.float)  # yapf: disable

        output = fitness_func(x)

        np.testing.assert_array_equal(output, [0., -0.25, -0.25, -1.])
