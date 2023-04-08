"""plots.py - Jérôme Leblanc

This file contains a plot class built on top of matplotlib to avoid redundancy
when plotting results.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Plot:
    """A class that handles plots related to the Mandelbrot class."""

    data_array: np.ndarray
    save_name: str
    extent: tuple[float, float, float, float]

    def __init__(self, nrows: int=1, ncols: int=1):
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
        """ Plots a heatmap.
        """

        if self.figure is None and self.axis is None:
            with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
                self.figure, self.axis = plt.subplots(1,1)

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            ax = plt.gca()

            im = ax.imshow(self.data_array, extent=self.extent,cmap=cmap)
            ax.set_xlabel("Re $c$")
            ax.set_ylabel("Im $c$")

            # Allows to correclty place and size the color bar.
            # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(im, cax=cax, label=cmap_label, extend=extend)

    def contour(self) -> None:
        """Plots the contour of a Mandelbrot maximum escape time.
        """

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
        x: list[float]
            List of x coordinates.
        y: list[float]
            List of y coordinates.

        """
        if self.figure is None:
            with plt.style.context("matplotlib_stylesheets/maps.mplstyle"):
                self.figure, self.axis = plt.subplots(1,1)

        with plt.style.context('matplotlib_stylesheets/maps.mplstyle'):
            self.axis.plot(x, y, markevery=[0], label=f"({x[0]}, {y[0]})")

    def norms(self, norms: list[float]) -> None:
        """ Produces plots of the norms of complex trajectories.
        """

        if self.figure is None:

            with plt.style.context("matplotlib_stylesheets/maps.mplstyle"):
                self.figure, self.axis = plt.subplots(ncols=self.ncols,
                                                      nrows=self.nrows,
                                                      sharey=True,
                                                      sharex=True,
                                                      figsize=(5,5))

                plt.xlabel("$n$")
                self.figure.text(-0.03, 0.5, '$|z_n|$', va='center', rotation='vertical')

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
        """ Load data arrays from files and parses the file's name into
        reusable data for the plots.
        """

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
