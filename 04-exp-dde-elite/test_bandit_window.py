import unittest
import numpy as np
from bandit_window import BanditWindow


class TestBanditWindow(unittest.TestCase):
    def test_add_result(self):
        np.random.seed(1)

        nb_choice = 2
        window_size = 3
        b = BanditWindow(nb_choice, window_size)

        np.testing.assert_array_equal(b._results, [])
        assert b._selections.shape == (1, nb_choice)

        res1 = 12
        s_idx1 = 0
        res2 = 7
        s_idx2 = 1
        res3 = 6
        s_idx3 = 0
        res4 = 2
        s_idx4 = 1

        b.add_result(res1, s_idx1)

        np.testing.assert_array_equal(b._results, [[12, 0]])
        np.testing.assert_array_equal(b._selections, [[0, 0], [1, 0]])

        b.add_result(res2, s_idx2)
        b.add_result(res3, s_idx3)
        b.add_result(res4, s_idx4)

        np.testing.assert_array_equal(b._results, [[0, 7], [6, 0], [0, 2]])
        np.testing.assert_array_equal(b._selections, [[0, 1], [1, 0], [0, 1]])

    def test_get_logits(self):
        np.random.seed(1)

        nb_choice = 2
        window_size = 3
        b = BanditWindow(nb_choice, window_size)

        res1 = 12
        s_idx1 = 0
        b.add_result(res1, s_idx1)

        logits = b.get_logits()
        probs = np.exp(logits - np.max(logits)) / np.sum(
            np.exp(logits - np.max(logits))
        )
        np.testing.assert_almost_equal(probs, [0.0, 1.0], 2)

        res2 = 7
        s_idx2 = 1
        res3 = 6
        s_idx3 = 0
        res4 = 9
        s_idx4 = 1
        b.add_result(res2, s_idx2)
        b.add_result(res3, s_idx3)
        b.add_result(res4, s_idx4)

        logits = b.get_logits()
        i = np.argmax(logits)

        assert i == 1
