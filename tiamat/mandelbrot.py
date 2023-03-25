"""
TODO HEADER
"""

import numba
import numpy as np
from numba import jit
from rich import print
from rich.progress import track


class Mandelbrot:
    """
    TODO Documentation
    """

    step: int
    domain: np.ndarray
    zn: np.ndarray
    escape_time: np.ndarray
    mask: np.ndarray
    #res:float # TODO Save the mask between iterations

    def __init__(self,
                 x_bounds: tuple[float, float],
                 y_bounds: tuple[float, float],
                 res: float):
        """
        TODO
        """

        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        #res = res

        X = np.arange(x_min, x_max, res)
        Y = np.arange(y_min, y_max, res)

        X, Y = np.meshgrid(X, Y)

        self.domain = X + Y * 1.j
        self.zn = self.domain.copy()

        #self.mask = squared_magnitude(self.zn) < 4
        self.mask = None
        self.cardioid_mask = None

        self.step = 0

        self.escape_time = None
        self.lyapunov = None


    def make_cardioid_mask(self) -> None:
        """
        TODO
        """
        # Main cardioid
        a=1/2
        imag = self.domain.imag
        reals_card = self.domain.real-1/4
        x2 = reals_card * reals_card
        y2 = imag * imag
        norm = x2 + y2
        LHS = (norm + a*reals_card)
        mask_card = (LHS * LHS > a*a*norm)

        # Main circle
        reals_circ = self.domain.real + 1
        norm_circ = reals_circ * reals_circ + imag * imag
        mask_circ = norm_circ > (1/16)


        self.cardioid_mask = (LHS * LHS > a*a*norm) * mask_circ
        #self.cardioid_mask =  mask_card



    def next_iteration(self, use_mask: bool) -> None:
        """Performs one iteration of the qaudratic map over the specified
        domain.

        Parameters
        ----------
        TODO
        """

        if use_mask:
            self.zn = np.where(self.mask*self.cardioid_mask, # Condition
                               quadratic_map(self.zn, self.domain), # If True
                               self.zn) # If false
            self.mask = squared_magnitude(self.zn) < 4
        else:
            self.zn = quadratic_map(self.zn, self.domain)

        self.step += 1


    def compute_escape_time(self, max_iteration_number: int) -> None:
        """Performs an equivalent version of the escape time algorithm.
        To do so, it uses `iterate` paired with `_escape_time_back` to check if
        an orbit as reached beyond the escape radius.

        Parameters
        ----------
        max_iteration_index : int
            The maximum number of iteration to perform.
        """
        self.mask = squared_magnitude(self.zn) < 4
        self.make_cardioid_mask()

        self.escape_time = np.zeros(self.domain.shape)

        for i in track(range(max_iteration_number), "Computing escape time..."):
            self.next_iteration(True)

            self.escape_time = np.where(self.mask*self.cardioid_mask, # Condition
                                        self.escape_time + 1, # If true
                                        self.escape_time) # If false

    # def save(self, path: str="./") -> None:

    #     params_string = f"x_{self.x_min}_{self.x_max}_y_{self.y_min}_{self.y_max}_res_{self.res}_step_{self.step}"

    #     if self.escape_time is None:
    #         saved_array = self.zn
    #     else:
    #         saved_array = np.concatenate((self.zn[np.newaxis, :],
    #                                       self.escape_time[np.newaxis,:]))

    #     np.save(f"{path}/mandelbrot_{params_string}", saved_array) # TODO Checker que ça existe


@jit(nopython=True)
def quadratic_map(z, c): # TODO Vectorize?
    """
    TODO Documentation
    """
    return z*z+c

@jit(nopython=True)
def squared_magnitude(z: np.ndarray) -> np.ndarray:
    """
    TODO DOCUMENTATION
    """
    x, y = z.real, z.imag
    return x * x + y * y


def orbit(start_point: np.complex128, num_iteration: int) -> np.ndarray: # WARNING Vérifier le data type requis
    Z = np.zeros(num_iteration, dtype=np.complex128)
    Z[0] = start_point

    for i in range(1, num_iteration):
        Z[i] = quadratic_map(Z[i-1], start_point)

    return Z
