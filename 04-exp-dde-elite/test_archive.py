from functions import behaviour_func, fitness_func
from archive import Archive
import unittest
import numpy as np


class TestArchive(unittest.TestCase):
    def test_sample(self):
        np.random.seed(1)
        archive = Archive(2)

        x = archive.sample(2)
        np.testing.assert_array_equal(x, [])

        elites = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.], ], dtype=np.float)  # yapf: disable
        archive.update(
            elites, np.arange(elites.shape[0]), -np.ones(elites.shape[0])
        )

        x = archive.sample(2)
        np.testing.assert_array_equal(x, [[1, 1], [1, 0]])

        x = archive.sample(5)
        np.testing.assert_array_equal(
            x, [
                [0., 0.],
                [1., 0.],
                [0., 1.],
                [1., 1.], ]
        )

    def test_compute_pos(self):
        np.random.seed(1)
        archive = Archive(2)

        elites = np.array([
            [0., 0.],                   # pos 59
            [np.pi, 0.],                # pos 50
            [np.pi, np.pi],             # pos 55
            [5 * np.pi / 4, 0],         # pos 11
            [np.pi / 2., 0.],           # pos 94
            [np.pi / 2., np.pi / 2],    # pos 61
        ])  # yapf: disabled
        bs, coords = behaviour_func(elites, archive.links)

        pos = archive.compute_pos(bs)

        np.testing.assert_array_equal(pos, [59, 50, 55, 11, 95, 72])

    def test_update(self):
        np.random.seed(1)
        archive = Archive(2)

        elites = np.array([
            [0., 0.],                   # pos 59
            [np.pi, 0.],                # pos 50
            [np.pi, np.pi],             # pos 55
            [5 * np.pi / 4, 0],         # pos 11
            [np.pi / 2., 0.],           # pos 94
            [np.pi / 2., np.pi / 2],    # pos 61
        ])  # yapf: disabled

        bs, coords = behaviour_func(elites, archive.links)
        pos = archive.compute_pos(bs)
        f = fitness_func(elites)
        successes = archive.update(elites, pos, f)

        assert successes == 6
        filtered_d = np.array(list(filter(lambda x: x is not None, archive.d)))
        assert filtered_d.shape[0] == 6
        assert archive.d[59] is not None
        assert archive.d[50] is not None
        assert archive.d[55] is not None
        assert archive.d[11] is not None
        assert archive.d[95] is not None
        assert archive.d[72] is not None

        successes = archive.update(elites, pos, f)

        assert successes == 0
        filtered_d = np.array(list(filter(lambda x: x is not None, archive.d)))
        assert filtered_d.shape[0] == 6
