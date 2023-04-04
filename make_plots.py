""" make_plots.py - Jérôme Leblanc

The scripts produces the plots needed for my project report.
"""

from glob import glob

import numpy as np
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
    # TODO

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



if __name__ == '__main__':
    console = Console()
    console.print("\nStart of program.", style="green")

    #escape_time_plots("./data/*escape_time_*y_0_1.5*_res*_step_100*")
    #escape_time_plots("./data/*escape_time_*y_0_0.05*res_0.0001*")
    orbit_plots("./data/mandelbrot_escape_time_x_-2_1_y_0_1.5_res_0.001_step_100.npy",
                [-.12-.75j, -0.5+0.56j, 0.275+0.525j, -1.3, 0.38-0.34j])
    #orbit_plots("./data/mandelbrot_escape_time_x_-2_1_y_0_1.5_res_0.001_step_100.npy",
    #            [])

    console.print("\nEnd of program.", style='green')