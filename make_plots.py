# TODO HEADER
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

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
        plot.heatmap()
        plot.save_plot(directory='./figures/')

if __name__ == '__main__':
    console = Console()
    console.print("\nStart of program.", style="green")

    escape_time_plots("./data/*escape_time_*res_0.001_step_*")

    console.print("\nEnd of program.", style='green')