""" make_plots.py - Jérôme Leblanc

The scripts produces the plots needed for my project report.
Its one of the last script I had to do before the project's deadline.
Please do not expect too much from this file!
"""

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.console import Console

from tiamat.mandelbrot import orbit
from tiamat.plot import Plot


def escape_time_plots(file_patern: str) -> None:
    """ This function produces the escape time plots.

    Parameters
    ----------
    file_patern: str
        The string patern to glob to fetch the desired files.
    """
    files = glob(file_patern)

    print("Plotting escape time figures for:")

    console = Console()
    for f in files:
        console.print(f)

        plot = Plot()

        y_coord = f.split("y_")[1].split("_")[0]

        if y_coord == "0":
            flip = True
        else:
            flip = False

        plot.load_data(f, flip=flip)
        plot.heatmap(cmap_label="Temps d'évasion", extend='max')
        plot.save_plot(directory='./figures/')


def orbit_plots(file: str, complex_start_points: list[complex]) -> None:
    """Produces the plot of the orbits with the countour of the Mandelbrot
    set."""

    print("Plotting orbit figures for:")
    console = Console()
    console.print(file)

    plot_map = Plot()
    plot_norm = Plot(nrows=len(complex_start_points))

    plot_map.load_data(file, flip=True)

    plot_map.contour()

    plot_map.axis.set_ylim(-1,1)
    plot_map.axis.set_xlim(-1.5,.5)

    orbits = np.array([orbit(c,10) for c in complex_start_points])

    for orb in orbits:
        plot_map.orbit(orb.real, orb.imag)
        plot_norm.norms(np.abs(orb))

    plot_map.legend()

    plot_map.save_plot(fname='orbits_map.pdf', directory='./figures/')

    plot_norm.save_plot(fname='orbits_norm.pdf', directory='./figures/')


def lyapunov_plot(file: str) -> None:
    """Plots the heatmap of the Lyapunov exponenet inside the Mandelbrot set."""

    print("Plotting lyapunov figures for:")
    console = Console()
    console.print(file)

    plot_f = Plot()

    plot_f.load_data(file, flip=True)

    with plt.style.context("matplotlib_stylesheets/maps.mplstyle"):
        plot_f.figure, plot_f.axis = plt.subplots(1,1)

        im = plot_f.axis.imshow(plot_f.data_array, extent=(-2, 1, -1.5, 1.5),
                   cmap='jet', origin='lower',vmin=-10, vmax=1)

        divider = make_axes_locatable(plot_f.axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax, label="$\lambda$",extend='both')

        plot_f.axis.set_xlim(-1.5, 0.5)
        plot_f.axis.set_ylim(-1, 1)
        plot_f.axis.set_ylabel("Im $c$")
        plot_f.axis.set_xlabel("Re $c$")
        plot_f.save_plot(fname='lyapunov.pdf', directory='./figures/')


def lyapunov_orbit_plot(file: str) -> None:
    """Plots the heatmap of the Lyapunov exponenet inside the Mandelbrot set.
    It also reduces the alpha of the colormap.
    +
    Plots orbits on top of the heatmap.
    """

    print("Plotting lyapunov + orbits figures for:")
    console = Console()
    console.print(file)

    plot_f = Plot()

    plot_f.load_data(file, flip=True)

    with plt.style.context("matplotlib_stylesheets/maps.mplstyle"):
        plot_f.figure, plot_f.axis = plt.subplots(1,1)

        im = plot_f.axis.imshow(plot_f.data_array, extent=(-2, 1, -1.5, 1.5),
                                alpha=0.5,
                                cmap='jet', origin='lower',vmin=-10, vmax=1)

        divider = make_axes_locatable(plot_f.axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax, label="$\lambda$",extend='both')

        plot_f.axis.set_xlim(-1.5, 0.5)
        plot_f.axis.set_ylim(-1, 1)
        plot_f.axis.set_ylabel("Im $c$")
        plot_f.axis.set_xlabel("Re $c$")

        #
        initial_points = (-1e-2/np.sqrt(2)+1e-2j/np.sqrt(2))
        scales = np.linspace(1, 70, num=20)
        points = np.array([orbit(initial_points*a, 4) for a in scales])
        for p in points:
            plot_f.axis.plot(p.real, p.imag, c='k', markevery=[0])
        #
        initial_points = (-1e-2/np.sqrt(2)+1e-2j/np.sqrt(2))
        scales = np.linspace(1, 25, num=10)
        points = np.array([orbit(initial_points*a-1, 2) for a in scales])
        for p in points:
           plot_f.axis.plot(p.real, p.imag, c='k', markevery=[0])

        plot_f.save_plot(fname='lyapunov_orbits.pdf', directory='./figures/')


if __name__ == '__main__':
    console = Console()
    console.print("\nStart of program.", style="green")

    #escape_time_plots("./data/*escape_time_*y_0_1.5*_res*_step_100*")

    #escape_time_plots("./data/*escape_time_*y_0_0.05*res_0.0001*")

    orbit_plots("./data/mandelbrot_escape_time_x_-2_1_y_0_1.5_res_0.001_step_100.npy",
                [-.12-.75j, -0.5+0.56j, 0.275+0.525j, -1.3, 0.38-0.34j])

    #lyapunov_orbit_plot("./data/mandelbrot_lyapunov_x_-2_1_y_0_1.5_res_0.001_step_100.npy")

    console.print("\nEnd of program.", style='green')