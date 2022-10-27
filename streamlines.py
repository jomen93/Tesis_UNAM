import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from Globals import * 
import read_binary
import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set latex configuration
mpl.rc("text", usetex=True)
mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

x_range = x_size/AU
y_range = y_size/AU

# 
dx = 60/256
# Creación del vector de posicion x
x = np.linspace(0,x_range,256)
# Creación del vector de posicion y
y = np.linspace(0,y_range,256)

# creacion del meshgrid
x, y = np.meshgrid(x, y)

# Obtención de los archivos de velocidad 
n=100
# file = "cuts/CutZ."+str(n).zfill(4)+".bin"
file = "CutZ."+str(n).zfill(4)+".bin"

data = read_binary.get_data(file)
# Velocidad en x
u = data[1]
# Velocidad en y
v = data[2]
# Velocidad en z
w = data[3]

fig = plt.figure(figsize=(5,5))
ax = plt.gca()
plt.streamplot(x, y, u.T, v.T, 
	color="k", 
	linewidth=0.3,
	density=10.0, 
	arrowstyle="-")
M = np.sqrt(u**2 + v**2)
# im = plt.quiver(x, y, u, v, M, cmap=plt.cm.jet, width=1/256)
vel = np.sqrt(u**2 + v**2 + w**2)
im = plt.imshow(vel.T, 
				cmap="jet", 
				extent=[0,x_range, 0,y_range], 
				origin="lower",
				norm=LogNorm()
				)
plt.axis("square")
plt.xlabel("x[AU]")
plt.ylabel("y[AU]")
plt.title("Densidad")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(im, extend="both", cax=cax)
plt.savefig("streamline_"+str(n)+".png", transparent=True)
plt.show()