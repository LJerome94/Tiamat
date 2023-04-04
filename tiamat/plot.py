import cmasher as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plot:
    # TODO

    data_array: np.ndarray
    save_name: str
    extent: tuple[float, float, float, float]

    def __init__(self, nrows: int=1, ncols: int=1):
        # TODO
        self.figure = None
        self.axis = None
        self.save_name = ''
        self.nrows = nrows
        self.ncols = ncols
        self.current_plot = 0


    def save_plot(self, fname: str='', directory: str='') -> None:
        """ Saves the plot to a given file name in .pdf format.

        Parameters
        ----------
        fname: str
            The name of the file to be saved. The class establishes one
            by default based on the imported data file name.
        directory:
            The directory to save the file.
        """

        if fname != '':
            self.save_name = fname

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            self.figure.savefig(fname=directory+self.save_name, dpi=1200)

    def heatmap(self, cmap='jet', cmap_label: str="", extend: str='none') -> None:
        # TODO
        if self.figure is None and self.axis is None:
            with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
                self.figure, self.axis = plt.subplots(1,1)

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            ax = plt.gca() # FIXME Changer pour les axes de la classe

            im = ax.imshow(self.data_array, extent=self.extent,cmap=cmap)
            ax.set_xlabel("Re $c$")
            ax.set_ylabel("Im $c$")

            # Allows to correclty place and size the color bar.
            # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(im, cax=cax, label=cmap_label, extend=extend)

    def contour(self) -> None:
        # TODO Documentation
        if self.figure is None and self.axis is None:
            with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
                self.figure, self.axis = plt.subplots(1,1)

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            self.save_name = self.save_name.replace("escape_time", "contour")

            self.axis.set_aspect('equal')

            plt.contour(1*(self.data_array==self.data_array.max()),
                        colors=['w', 'k'],
                        extent=self.extent,
                        levels=1)

            plt.xlabel("Re $c$")
            plt.ylabel("Im $c$")

    def orbit(self, x: list[float], y: list[float]) -> None:
        """Plots the orbit on the complex plane given a list of coordinates.

        Parameters
        ----------
        # TODO

        """
        if self.figure is None:
            with plt.style.context("matplotlib_stylesheets/maps.mplstyle"):
                self.figure, self.axis = plt.subplots(1,1)

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            self.axis.plot(x, y, markevery=[0], label=f"({x[0]}, {y[0]})")

    def norms(self, norms: list[float]) -> None:
        # TODO DOCS

        if self.figure is None:

            with plt.style.context("matplotlib_stylesheets/maps.mplstyle"):
                self.figure, self.axis = plt.subplots(ncols=self.ncols,
                                                      nrows=self.nrows,
                                                      sharey=True,
                                                      sharex=True,
                                                      figsize=(5,5))

                plt.xlabel("$n$")
                self.figure.text(-0.03, 0.5, '$|z_n|^2$', va='center', rotation='vertical')

        with plt.style.context("matplotlib_stylesheets/maps.mplstyle"):
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            self.axis[self.current_plot].plot(np.arange(len(norms)),
                                              norms,
                                              c=colors[self.current_plot])
        self.current_plot += 1



    def legend(self) -> None:
        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            self.axis.legend(loc=0)

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
