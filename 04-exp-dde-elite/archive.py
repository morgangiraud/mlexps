import numpy as np
from typing import Callable, List, Tuple, Optional
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plot
matplotlib.use("Agg")


class Archive:
    def __init__(self, nb_links: int, nb_bins=100):
        check_pow2 = round(nb_bins**0.5)**2 == nb_bins
        assert check_pow2, 'nb_bins must be a power of two'

        self.nb_links: int = nb_links  # Phenotype dimensions
        self.nb_bins: int = nb_bins
        self.links: np.ndarray = np.ones(self.nb_links)
        self.r_max: int = np.sum(self.links)

        self.d: List[Optional[Tuple[np.ndarray, float]]] = [None] * self.nb_bins

    def init(self, behaviour_f: Callable, fitness: Callable) -> None:
        # A simple way to init to extreme possible solutions
        # theta0 = np.linspace(0, 2 * np.pi, num=100, endpoint=False)
        # theta0 = theta0[:, np.newaxis]
        # elites = np.concatenate(
        #     (theta0, np.zeros([100, self.nb_links - 1])),
        #     axis=1
        # )
        elites = np.random.standard_normal([100, self.nb_links])
        bs = behaviour_f(elites, self.links)
        ps = self.compute_pos(bs)
        fs = fitness(elites)

        self.update(elites, ps, fs)

    def get_elites(self) -> np.ndarray:
        filtered_d = [x for x in self.d if x is not None]
        d = np.array([x[0] for x in filtered_d])

        return d

    def compute_pos(self, coords: np.ndarray) -> np.ndarray:
        # Quantize
        coords_norm = np.linalg.norm(coords, axis=1, keepdims=True)
        coef = np.minimum(coords_norm, self.r_max) / (coords_norm + 1e-9)
        bin_limit = 2 * self.r_max / self.nb_bins**0.5
        coords = coords * coef + self.r_max
        coords = np.array(np.floor(coords / bin_limit))

        # Get positions
        # y-axis -> lines
        # x-axis -> columns
        pos = coords[:, 1] * self.nb_bins**0.5 + coords[:, 0]

        return pos.astype(np.int)

    def sample(self, nb_samples: int) -> np.ndarray:
        d = self.get_elites()
        if d.shape[0] == 0:
            return np.array([])

        nb_samples = min(d.shape[0], nb_samples)
        idx = np.random.choice(
            np.arange(d.shape[0]), size=nb_samples, replace=False
        )

        return d[idx]

    def update(
        self, children: np.ndarray, ps: np.ndarray, fs: np.ndarray
    ) -> int:
        successes = 0

        for i in range(children.shape[0]):
            new_elite = children[i]
            p = ps[i]
            f = fs[i]
            current_elite = self.d[p]
            if current_elite is None:
                self.d[p] = (new_elite, f)
                successes += 1
            else:
                if current_elite[1] < f:
                    self.d[p] = (new_elite, f)
                    successes += 1

        return successes

    def draw_illuminated_map(self, fullpath: str) -> None:
        clean_min_fit = [x[1] if x is not None else np.nan for x in self.d]
        d_fit_reshaped = np.reshape(
            clean_min_fit, [int(self.nb_bins**0.5), int(self.nb_bins**0.5)]
        )
        final_d = np.flip(d_fit_reshaped, 0)

        # Vmin is -0.5 because:
        # - Found solutions are always better than 0.5
        # - It create a more understandable visual
        ax = sns.heatmap(final_d, linewidth=0.5, vmin=-.5, vmax=0)
        ax.get_figure().savefig(fullpath)
        plot.clf()
