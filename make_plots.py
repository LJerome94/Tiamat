# TODO HEADER
import matplotlib.pyplot as plt
import numpy as np

from tiamat.plot import Plot

if __name__ == '__main__':
    #plot = Plot()

    M = np.load("./data/mandelbrot_escape_time_x_-2_1_y_0_1.5_res_0.001_step_100.npy", allow_pickle=False)

    #plot.escape_time(M)

    #plot.save()

    fig = plt.figure()
    plt.imshow(M,cmap='jet')
    fig.savefig('plot_local.png', dpi=1200)
