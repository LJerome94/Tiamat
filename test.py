import glob as glob
import time as time

#plt.style.use("./style.mplstyle")
import cmasher as cm
import matplotlib.pyplot as plt
import numpy as np
from rich import print

from tiamat import mandelbrot as mb
from tiamat.plot import Plot

#M = mb.Mandelbrot((-0.75-.3,-.5),(0.05,2*.325),0.001)

#t0 = time.time()
M = mb.Mandelbrot((-2,0.5),(0,1.3),0.001)
M.compute_lyapunov(30)
#t1 = time.time()


fig = plt.figure()
#f = glob.glob("*50.txt")[0]
#E = np.load(f)
#plt.imshow(M.escape_time,cmap='jet', origin='lower')
plt.imshow(M.lyapunov,cmap='gnuplot', origin='lower')
#print(E.shape)

#plt.imshow(np.concatenate((np.flip(E,axis=0), E)),
#           cmap=cm.horizon,
#           extent=[-2,1,-1.5,1.5],
#           origin='lower')
plt.colorbar()

#plt.plot(Z.real, Z.imag)

#plt.axis("off")
fig.savefig("lyapunov.png",dpi=1200)

#p = Plot()

#p.escape_time(E)

#p.save()

#print(f"End of program: [yellow]{t1-t0}")
