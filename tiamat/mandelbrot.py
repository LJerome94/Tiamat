"""
TODO HEADER
"""

import numpy as np
from rich.progress import track


class Mandelbrot:
    """
    TODO Documentation
    """

    step: int
    domain: np.ndarray
    zn: np.ndarray
    escape_time: np.ndarray
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

        res = res

        X = np.arange(x_min, x_max, res)
        Y = np.arange(y_min, y_max, res)

        X, Y = np.meshgrid(X, Y)

        self.domain = X + Y * 1.j
        self.zn = self.domain.copy()

        self.mask = self.squared_magnitude() < 4

        self.step = 0

        self.escape_time = None


    def next_iteration(self, use_mask: bool) -> None:
        """Performs one iteration of the qaudratic map over the specified
        domain.

        Parameters
        ----------
        TODO
        """

        if use_mask:
            #mask = self.squared_magnitude() < 4

            self.zn = np.where(self.mask, # Condition
                               quadratic_map(self.zn, self.domain), # If True
                               self.zn) # If false
            self.mask = self.squared_magnitude() < 4
        else:
            self.zn = self.quadratic_map(self.zn, self.domain)

        self.step += 1


    def quadratic_map(self, z:np.ndarray, c:np.ndarray) -> np.ndarray:
        return z * z + c # WARNING


    def compute_escape_time(self, max_iteration_number: int) -> None:
        """Performs an equivalent version of the escape time algorithm.
        To do so, it uses `iterate` paired with `_escape_time_back` to check if
        an orbit as reached beyond the escape radius.

        Parameters
        ----------
        max_iteration_index : int
            The maximum number of iteration to perform.
        """

        self.escape_time = np.zeros(self.domain.shape)

        for i in track(range(max_iteration_number)):
            self.next_iteration(True)

            self.escape_time = np.where(self.mask, # Condition
                                        self.escape_time + 1, # If true
                                        self.escape_time) # If false

    # def compute_escape_time_real(self, N):

    #     x = self.zn.real
    #     y = self.zn.imag

    #     self.escape_time = np.zeros(x.shape)

    #     for i in track(range(N)):
    #         mask = x * x + y * y < 4
    #         #self.X = np.where(mask,
    #         #                    x * x - y * y + self.Cx,
    #         #                    x)
    #         #self.Y = np.where(mask,
    #         #                    2 * x * y + self.Cy,
    #         #                    y)

    #         zn = np.where(mask,
    #                       x * x - y * y + 2j * x * y,
    #                       self.zn)

    #         x = zn.real
    #         y = zn.imag

    #         self.zn = zn

    #         mask = x * x + y * y < 4
    #         self.escape_time = np.where(mask,
    #                                     self.escape_time+1,
    #                                     self.escape_time)


    def squared_magnitude(self) -> np.ndarray:
        """
        TODO DOCUMENTATION
        """

        x, y = self.zn.real, self.zn.imag

        return x * x + y * y


    # def save(self, path: str="./") -> None:

    #     params_string = f"x_{self.x_min}_{self.x_max}_y_{self.y_min}_{self.y_max}_res_{self.res}_step_{self.step}"

    #     if self.escape_time is None:
    #         saved_array = self.zn
    #     else:
    #         saved_array = np.concatenate((self.zn[np.newaxis, :],
    #                                       self.escape_time[np.newaxis,:]))

    #     np.save(f"{path}/mandelbrot_{params_string}", saved_array) # TODO Checker que ça existe



def quadratic_map(z, c):
    return z*z+c


def orbit(start_point: np.complex128, num_iteration: int) -> np.ndarray: # WARNING Vérifier le data type requis
    Z = np.zeros(num_iteration, dtype=np.complex128)
    Z[0] = start_point

    for i in range(1, num_iteration):
        Z[i] = quadratic_map(Z[i-1], start_point)

    return Z
