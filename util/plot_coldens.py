import read_binary
from Globals import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import cm
import matplotlib.animation as animation

# Set latex configuration
mpl.rc("text", usetex=True)
mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

fig = plt.figure()


# path = "coldens_initial_state/"
# path = "coldens_X_30/"
# path = "coldens_X_60/"
path = "coldens_X_90/"

n = 202
file_xray = "XrayX."+str(n).zfill(4)+".bin"
print()
data_xray = read_binary.get_data_xray(path+file_xray)
data_xray = np.transpose(data_xray, [1,0])

cmap = cm.get_cmap("viridis")

name = "density_coldens"+str(n).zfill(4)

fileout = name+".png"

im = plt.imshow(data_xray, 
				cmap=cmap, 
				extent=[0,x_size/AU, 0,y_size/AU], 
				origin="lower",
				interpolation="Nearest",
				norm=LogNorm(vmin=1e-4,vmax=1e0),
				)
plt.xlabel("x[AU]")
plt.ylabel("y[AU]")
plt.title("Xray map")
plt.colorbar(im, extend="both")
plt.savefig(path+fileout, transparent=True)
plt.show()


### Animation
fig = plt.figure()
ims = []
for n in range(1, 220):
	file_xray = "XrayX."+str(n).zfill(4)+".bin"
	print(file_xray)
	data_xray = read_binary.get_data_xray(path+file_xray)
	data_xray = np.transpose(data_xray, [1,0])
	im = plt.imshow(data_xray, 
				cmap=cmap, 
				extent=[0,x_size/AU, 0,y_size/AU], 
				origin="lower",
				# interpolation="Nearest",
				norm=LogNorm(vmin=1e-4,vmax=1e0),
				)
	plt.xlabel("x[AU]")
	plt.ylabel("y[AU]")
	plt.title("Columnar density")
	ims.append([im])

clb = fig.colorbar(im, extend="both")
# clb.set_label('label', labelpad=-40, y=1.05, rotation=0)
clb.ax.set_title('$\\rho$')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)
ani.save(path+file_xray[:-4]+'.mp4', writer=writer)
plt.show()

