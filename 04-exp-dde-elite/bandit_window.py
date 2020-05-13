import numpy as np


class BanditWindow(object):
    def __init__(self, nb_choice, window_size):
        self._nb_choice = nb_choice
        self._window_size = window_size
        self._results = np.array([])
        self._selections = np.array([[0] * nb_choice])

        self._successes = np.array([[0] * nb_choice])
        self._total_selections = np.array([[0] * nb_choice])

        self._epsilon = 1e-11

    def add_result(self, result: float, selection_idx: int):
        r = np.zeros((1, self._nb_choice))
        r[0, selection_idx] = result
        if self._results.shape[0] == 0:
            self._results = r
        elif self._results.shape[0] >= self._window_size:
            self._results = np.concatenate([self._results[1:, :], r], axis=0)
        else:
            self._results = np.concatenate([self._results, r], axis=0)

        s = np.zeros((1, self._nb_choice))
        s[0, selection_idx] = 1
        if self._selections.shape[0] >= self._window_size:
            self._selections = np.concatenate([self._selections[1:, :], s], axis=0)
        else:
            self._selections = np.concatenate([self._selections, s], axis=0)

        self._successes = np.sum(self._results, axis=0)
        self._total_selections = np.sum(self._selections, axis=0)

    def get_logits(self):
        t = np.sum(self._successes) + 1

        q_a = self._successes / (self._total_selections + self._epsilon)
        r = np.sqrt(2 * np.log(t) / (self._total_selections + self._epsilon))

        logits = q_a + r

        return logits

    def __str__(self):
        return "successes: {}, selections: {}".format(self._successes, self._total_selections)
