import matplotlib.pyplot as plt
import numpy as np

from tiamat import mandelbrot as mb


class Plot:
    # TODO

    data_array: np.ndarray
    save_name: str

    def __init__(self):
        # TODO
        self.figure = None
        self.axis = None
        self.save_name = ''

    def save_plot(self, fname: str='', directory: str='') -> None:
        # TODO

        if self.save_name == '' and fname != '':
            self.save_name = fname

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            self.figure.savefig(fname=directory+self.save_name, dpi=1200, format='pdf')

    def heatmap(self) -> None:
        # TODO
        if self.figure is None and self.axis is None:
            with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
                self.figure, self.axis = plt.subplots(1,1)

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'): # WARNING PAS NECESSAIRE
            plt.imshow(self.data_array, cmap='jet')
            plt.colorbar()

    def load_data(self, fname: str, flip: bool=False) -> None:
        # TODO
        self.data_array = np.load(fname, allow_pickle=False)

        self.save_name = fname.split("mandelbrot_")[1].split(".npy")[0]
        self.save_name += '.pdf'

        if flip:
            self.data_array = np.concatenate((np.flip(self.data_array,axis=0),
                                              self.data_array))