import time as time

import numpy as np
from numba import njit
from rich import print

from tiamat.mandelbrot import quadratic_map, squared_magnitude


@njit(parallel=True)
def mandelbrot(X, Y, n_iteration):
    """
    TODO Documentation
    """

    domain = X + Y * 1.j
    zn = domain.copy()

    mask = squared_magnitude(zn) < 4

    escape_time = np.zeros(domain.shape)

    for i in range(n_iteration):
        zn = np.where(mask, # Condition
                        quadratic_map(zn, domain), # If True
                        zn) # If false
        mask = squared_magnitude(zn) < 4


        escape_time = np.where(mask, # Condition
                                    escape_time + 1, # If true
                                    escape_time) # If false


if __name__ == "__main__":
    t0 = time.time()
    X = np.arange(-2, 1, 0.001)
    Y = np.arange(0, 1.5, 0.001)

    X, Y = np.meshgrid(X, Y)

    mandelbrot(X, Y, 100)

    print(f"End of program: {time.time()-t0}")
