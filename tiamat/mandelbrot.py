""" mandelbrot.py - Jérôme Leblanc

This files contains the class and functions required to compute various values
and trajectories of the Mandelbrot set.
"""

import numpy as np
from numba import jit
from rich.progress import track


class Mandelbrot:
    """ A class that contains Mandelbrot set related values such as its domain,
    escape time, and Lyapunov exponents
    """

    step: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    res: float
    domain: np.ndarray
    zn: np.ndarray
    escape_time: np.ndarray
    mask: np.ndarray
    lyapunov: np.ndarray

    def __init__(self,
                 x_bounds: tuple[float, float],
                 y_bounds: tuple[float, float],
                 res: float):
        """Constructs a Mandelbrot object given the provided complex domain.

        Paramters
        ---------
        x_bounds : tuple[float]
            The bounds of the real parts of the domain C.
        y_bounds : tuple[float]
            The bounds of the imaginary parts of the domain C.
        res: float
            The resolution of the domain. It corresponds to the spacing between
            pixels both on x and y axies.
        """

        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds

        self.res = res

        X = np.arange(self.x_min, self.x_max, res)
        Y = np.arange(self.y_min, self.y_max, res)

        X, Y = np.meshgrid(X, Y)

        self.domain = X + Y * 1.j
        self.zn = self.domain.copy()

        self.mask = None
        self.cardioid_mask = None

        self.step = 0

        self.escape_time = None
        self.lyapunov = None


    def make_cardioid_mask(self) -> None:
        """ This creates a mask array the same size as `self.domain` to avoid
        computing values inside the main cardioid and the 1/2 bulb.
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

        # Main bulb
        reals_circ = self.domain.real + 1
        norm_circ = reals_circ * reals_circ + imag * imag
        mask_circ = norm_circ > (1/16)


        self.cardioid_mask = mask_card * mask_circ


    def next_iteration(self, use_mask: str) -> None:
        """Performs one iteration of the qaudratic map over the specified
        domain.

        Parameters
        ----------
        use_mask: bool
            Whether to use logic masks on the involved arrays in order to avoid
            some redundant computations.
        """

        if use_mask == 'both':
            self.zn = np.where(self.mask*self.cardioid_mask, # Condition
                               quadratic_map(self.zn, self.domain), # If True
                               self.zn) # If false
            self.mask = squared_magnitude(self.zn) < 4
        elif use_mask == 'norm':
            self.zn = np.where(self.mask, # Condition
                               quadratic_map(self.zn, self.domain), # If True
                               self.zn) # If false
            self.mask = squared_magnitude(self.zn) < 4
        elif use_mask == 'none':
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

        self.escape_time = np.zeros(self.domain.shape, dtype=int)
        self.step = 0

        progress_str = f"Computing escape time for {max_iteration_number} iterations..."
        for i in track(range(max_iteration_number), progress_str):
            self.next_iteration('both')

            self.escape_time = np.where(self.mask*self.cardioid_mask, # Condition
                                        self.escape_time + 1, # If true
                                        self.escape_time) # If false

        # Add maximum values in the cardioid and the first bulb
        self.escape_time = np.where(self.cardioid_mask, # Condition
                                    self.escape_time, # If true
                                    max_iteration_number) # If false

    def compute_lyapunov(self, max_iteration_number) -> None:
        """ Computes the Lyapunov exponents inside the Mandelbrot set.

        Parameters
        ----------
        max_iteration_number: int
            The number of iterations to perform in order to compute the Lyapunov
            exponents.
        """

        self.step = 0

        self.lyapunov = np.log(2*squared_magnitude(self.zn))

        self.mask = squared_magnitude(self.zn) < 4
        self.make_cardioid_mask()

        progress_str = f"Computing lyapunov exponents for {max_iteration_number} iterations..."
        for i in track(range(max_iteration_number), progress_str):
            self.next_iteration('norm')

            self.lyapunov = np.where(self.mask, # Condition
                                        self.lyapunov + np.log(squared_magnitude(2*self.zn)), # If true
                                        np.inf) # If false

        self.lyapunov /= max_iteration_number


    def save(self, attribute: str, path: str="./") -> None:
        """ Saves the specified attribute as a .npy file.

        Parameters
        ----------
        attribute: str
            The name of the class attribute to save.
        path: str
            The path where to save the file. Defaults to './'.
        """

        params_string = f"x_{self.x_min}_{self.x_max}_y_{self.y_min}_{self.y_max}_res_{self.res}_step_{self.step}"

        saved_array = getattr(self, attribute)

        np.save(f"{path}/mandelbrot_{attribute}_{params_string}", saved_array, allow_pickle=False)


@jit(nopython=True)
def quadratic_map(z, c):
    """ Computes the quadratic map f(z, c) = z^2 + c.

    Parameters
    ----------
    z: numpy.ndarray
        Array of complex values.
    c: numpy.ndarray
        Array of complex values.

    Returns
    -------
    out: numpy.ndarray
        z^2 + c
    """
    return z*z+c

@jit(nopython=True)
def squared_magnitude(z: np.ndarray) -> np.ndarray:
    """Compute the squared magnitudes of complex values inside an array.
    Optimized with Numba.

    Parameters
    ----------
    z: numpy.ndarray
        Array of complex values.

    Returns
    -------
    out: numpy.ndarray
        Array of the squared magnitudes.

    """
    x, y = z.real, z.imag
    return x * x + y * y


def orbit(start_point: np.complex128, num_iteration: int) -> np.ndarray:
    """ Computes the orbit of a given point under the Mandelbrot quadratic map
    for a given number of iterations.

    Parameters
    ----------
    start_point: numpy.complex128
        The starting point of the orbit
    num_itertaions: int

    Returns
    -------
    out: numpy.ndarray
        The points of the trajectory. It is a complex valued 1d array of length
        `num_iteration`.
    """
    Z = np.zeros(num_iteration, dtype=np.complex128)
    Z[0] = start_point

    for i in range(1, num_iteration):
        Z[i] = quadratic_map(Z[i-1], start_point)

    return Z
