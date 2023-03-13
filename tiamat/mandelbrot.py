import numpy as np


def quadratic_map(z, c):
    return z*z+c


class Mandelbrot:

    step: int
    domain: np.ndarray
    zn: np.ndarray
    escape_radius: float
    escape_time: np.ndarray
    res:float

    def __init__(self,
                 x_bounds: tuple[float, float],
                 y_bounds: tuple[float, float],
                 res: float,
                 escape_radius: float=2):
        """
        TODO
        """

        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds

        self.res = res

        X = np.arange(self.x_min, self.x_max, self.res)
        Y = np.arange(self.y_min, self.y_max, self.res)

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
        """Iterates the quadratic map over the specified domain.

        Parameters
        ----------
        TODO
        """
        for i in range(iteration_index):
            self.next_iteration(use_mask, mask_region)

            if callback != None:
                self._escape_time_callback()


    def next_iteration(self,
                       use_mask=False,
                       mask_region='inner') -> None:
        """Performs one iteration of the qaudratic map over the specified
        domain.

        Parameters
        ----------
        TODO
        """

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

        self.escape_time = np.zeros(self.domain.shape)

        self.iterate(max_iteration_number,
                     use_mask=True,
                     callback=self._escape_time_callback)


    def _escape_time_callback(self) -> None:
        """
        TODO
        """

        self.escape_time = np.where(np.abs(self.zn)<2, # Condition
                                    self.escape_time+1, # If true
                                    self.escape_time) # If false


    def save(self, path: str="./") -> None:
        """
        TODO
        """

        params_string = f"x_{self.x_min}_{self.x_max}_y_{self.y_min}_{self.y_max}_res_{self.res}_step_{self.step}"

        if self.escape_time is None:
            saved_array = self.zn
        else:
            saved_array = np.concatenate((self.zn[np.newaxis, :],
                                          self.escape_time[np.newaxis,:]))

        np.save(f"{path}/mandelbrot_{params_string}", saved_array) # TODO Checker que ça existe


def orbit(start_point: np.complex128, num_iteration: int) -> np.ndarray: # WARNING Vérifier le data type requis
    Z = np.zeros(num_iteration, dtype=np.complex128)
    Z[0] = start_point

    for i in range(1, num_iteration):
        Z[i] = quadratic_map(Z[i-1], start_point)

    return Z
