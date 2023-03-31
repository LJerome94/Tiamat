import cmasher as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plot:
    # TODO

    data_array: np.ndarray
    save_name: str
    extent: tuple[float, float, float, float]

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
            self.figure.savefig(fname=directory+self.save_name, dpi=100) # WARNING CHANGER LE DPI

    def heatmap(self, cmap='jet', cmap_label: str="") -> None:
        # TODO
        if self.figure is None and self.axis is None:
            with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
                self.figure, self.axis = plt.subplots(1,1)

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'): # WARNING PAS NECESSAIRE?
            ax = plt.gca()

            im = ax.imshow(self.data_array, extent=self.extent,cmap=cmap)
            ax.set_xlabel("Re $c$")
            ax.set_ylabel("Im $c$")

            # Allows to correclty place and size the color bar.
            # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(im, cax=cax,label=cmap_label)

    def contour(self) -> None:
        # TODO Documentation
        if self.figure is None and self.axis is None:
            with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
                self.figure, self.axis = plt.subplots(1,1)

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            self.save_name = self.save_name.replace("escape_time", "contour")

            plt.contour(1*(self.data_array==self.data_array.max()),
                        colors=['w', 'k'],
                        extent=self.extent,
                        levels=1)

            plt.xlabel("Re $c$")
            plt.xlabel("Im $c$")

    def orbit(self, x: list[float], y: list[float]) -> None:
        """Plots the orbit on the complex plane given a list of coordinates.

        Parameters
        ----------
        # TODO

        """
        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            plt.plot(x, y)


    def load_data(self, fname: str, flip: bool=False) -> None:
        # TODO
        self.data_array = np.load(fname, allow_pickle=False)

        self.save_name = fname.split("mandelbrot_")[1].split(".npy")[0]

        S = self.save_name.split("x_")[1].split("_")

        if flip==True:
            S[3] = '-' + S[4]

        self.extent = (float(S[0]),
                       float(S[1]),
                       float(S[3]),
                       float(S[4]))

        self.save_name += '.pdf'

        if flip:
            self.data_array = np.concatenate((np.flip(self.data_array,axis=0),
                                              self.data_array))