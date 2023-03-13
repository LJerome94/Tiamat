import time as time

t0 = time.time()
import matplotlib.pyplot as plt

plt.style.use("./style.mplstyle")
import numpy as np

from tiamat import mandelbrot as mb

#M = mandelbrot.Mandelbrot((-0.75-.4,0-.4),(-.325+0.2,.325+0.2),0.001)
M = mb.Mandelbrot((-2,1),(-1.5,1.5),0.001)

#M.iterate(30, use_mask=True)
M.compute_escape_time(30)

#Z = mb.orbit(0.27+0.55j,20)

fig = plt.figure()
plt.imshow(M.escape_time, cmap='inferno')

#plt.contour(M.domain.real, M.domain.imag, 1*(np.abs(M.zn)<2), cmap='binary')

#plt.plot(Z.real, Z.imag)

#plt.axis("off")
fig.savefig("test_zoom.png")
print(f"{time.time()-t0}")