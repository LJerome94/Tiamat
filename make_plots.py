# TODO HEADER
from glob import glob

#import matplotlib.pyplot as plt
#import numpy as np
from rich.console import Console

from tiamat.mandelbrot import orbit
from tiamat.plot import Plot


def escape_time_plots(file_patern: str) -> None:
    # TODO
    files = glob(file_patern)

    print("Plotting escape time figures for:")

    console = Console()
    for f in files:
        console.print(f)

        plot = Plot()
        plot.load_data(f, flip=True)
        plot.heatmap(cmap_label="Temps d'évasion")  # TODO checker pour si faut flipper
        plot.save_plot(directory='./figures/')

def orbit_plots(file: str, complex_start_points: list[complex]) -> None:
    # TODO

    print("Plotting orbit figure for:")
    console = Console()
    console.print(file)

    plot = Plot()
    plot.load_data(file, flip=True) # TODO checker pour si faut flipper

    plot.contour()

    for c in complex_start_points:
        orb = orbit(c,5)
        #print(orb)
        #exit()
        plot.orbit(orb.real, orb.imag)


    plot.save_plot(directory='./figures/')


if __name__ == '__main__':
    console = Console()
    console.print("\nStart of program.", style="green")

    #escape_time_plots("./data/*escape_time_*y_0_1.5*_res*_step_100*")
    orbit_plots("./data/mandelbrot_escape_time_x_-2_1_y_0_1.5_res_0.001_step_100.npy",
                [-.1+.7j])

    console.print("\nEnd of program.", style='green')