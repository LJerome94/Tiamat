import numpy as np


def quadratic_map(z, c):
    return z*z+c


class Mandelbrot:

    step: int
    domain: np.ndarray
    zn: np.ndarray
    escape_radius: float
    escape_time: np.ndarray

    def __init__(self,
                 x_bounds: tuple[float, float],
                 y_bounds: tuple[float, float],
                 res: float,
                 escape_radius: float=2):

        x_min = x_bounds[0]
        x_max = x_bounds[1]

        y_min = y_bounds[0]
        y_max = y_bounds[1]

        X = np.arange(x_min, x_max, res)
        Y = np.arange(y_min, y_max, res)

        X, Y = np.meshgrid(X,Y)

        self.domain = X + Y * 1.j
        self.zn = self.domain.copy()

        self.escape_radius = escape_radius

        self.step = 0

        self.escape_time = None

    def iterate(self,
                iteration_index,
                use_mask=False,
                mask_region='inner',
                callback=None)-> None:
        for i in range(iteration_index):
            self.next_iteration(use_mask, mask_region)

            if callback != None:
                pass # TODO Ajouter le callback

    def next_iteration(self, use_mask=False, mask_region='inner') -> None:

        if use_mask:
            if mask_region == 'inner':
                mask = np.abs(self.zn) < self.escape_radius
            elif mask_region == 'outer':
                mas = np.abs(self.zn) > self.escape_radius
            else:
                pass # TODO RAISE ERROR AND ADD CUSTOM MASK

            self.zn = np.where(mask,quadratic_map(self.zn, self.domain), self.zn)
        else:
            self.zn = quadratic_map(self.zn, self.domain)

    def escape_time(self, max_iteration_index) -> None:
        self.iterate(max_iteration_index, use_mask=True, callback=self._escape_time_callback)

    def _escape_time_callback(self):
        self.escape_time = np.where(np.abs(self.zn)<2, self.escape_time+1, self.escape_time) # TODO Tester ça ici
