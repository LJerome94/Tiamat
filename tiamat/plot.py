import matplotlib.pyplot as plt
import numpy as np

from tiamat import mandelbrot as mb


class Plot:

    def __init__(self):
        n_rows=1
        n_cols=1
        #with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
        #    self.figure, self.axis = plt.subplots(nrows=n_rows, ncols=n_cols) # TODO Move this elsewhere

        self.figure = None
        self.axis = None

    def save(self, fname: str='./plot.png') -> None:
        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            self.figure.savefig(fname=fname, dpi=1200)

    def escape_time(self, array: np.ndarray) -> None:
        if self.figure is None and self.axis is None:
            with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
                self.figure, self.axis = plt.subplots(1,1)

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            plt.imshow(array, cmap='jet')
            plt.colorbar()