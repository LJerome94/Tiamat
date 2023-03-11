import numpy as np


def quadratic_map(z, c):
    return z*z+c

class Mandelbrot:

    def __init__(self, x_bounds, y_bounds, x_res: int, y_res: int, escape_radius=2):

        x_min = x_bounds[0]
        x_max = x_bounds[1]

        y_min = y_bounds[0]
        y_max = y_bounds[1]

        X = np.linspace(x_min, x_max,  x_res)
        Y = np.linspace(y_min, y_max,y_res)

        X, Y = np.meshgrid(X,Y)

        self.domain = X + Y * 1.j
        self.zn = self.domain.copy()

        self.escape_radius = escape_radius

    def next_iteration(self, use_mask:bool=False, mask_region='inner'):

        if use_mask:
            if mask_region == 'inner':
                mask = np.abs(self.zn) < 2
            elif mask_region == 'outer':
                mas = np.abs(self.zn) > 2
            else:
                pass # TODO RAISE ERROR AND ADD CUSTOM MAX

            self.zn = np.where(mask,quadratic_map(self.zn, self.domain), self.zn)
        else:
            self.zn = quadratic_map(self.zn, self.domain)

