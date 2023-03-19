import time as time

#plt.style.use("./style.mplstyle")
import cmasher as cm
import matplotlib.pyplot as plt
import numpy as np
from rich import print

from tiamat import mandelbrot as mb
from tiamat.plot import Plot

#M = mandelbrot.Mandelbrot((-0.75-.4,0-.4),(-.325+0.2,.325+0.2),0.001)

t0 = time.time()
M = mb.Mandelbrot((-2,1),(0,1.5),0.001)
M.compute_escape_time(100)
t1 = time.time()
#M.save()
#Z = mb.orbit(0.27+0.55j,20)

#fig = plt.figure()
#plt.imshow(M.escape_time, cmap=cm.horizon)
#plt.colorbar()

#plt.contour(M.domain.real, M.domain.imag, 1*(np.abs(M.zn)<2), cmap='binary')

#plt.plot(Z.real, Z.imag)

#plt.axis("off")
#fig.savefig("new_meshgrid.png", dpi=1200)

#p = Plot()

#p.plot_test()

#p.save()

print(f"End of program: [yellow]{t1-t0}")
