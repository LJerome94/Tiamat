import matplotlib.pyplot as plt
import numpy as np

from tiamat import mandelbrot as mb


class Plot:

    mandelbrot_object: mb.Mandelbrot

    def __init__(self):
        n_rows=1
        n_cols=1
        #with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
        #    self.figure, self.axis = plt.subplots(nrows=n_rows, ncols=n_cols) # TODO Move this elsewhere

        self.figure = None
        self.axis = None

    def save(self, directory: str='./plot.png') -> None:
        self.figure.savefig(directory)

    def plot_test(self):
        #with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
        self.axis.plot([1,2],[3,7])

    def escape_time(self, directory) -> None:
        if self.figure is None and self.axis is None:
            with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
                self.figure, self.axis = plt.subplots(1,1)

        escape_data = np.genfromtxt(directory)
        plt.imshow(escape_data)

def load_array(directory: str) -> np.ndarray: # TODO Nécessaire?
    return np.genfromtxt(directory)
