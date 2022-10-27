import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from Globals import *
import numpy as np
import read_binary
import os

# Set latex configuration
mpl.rc("text", usetex=True)
mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

fig = plt.figure()

# rotation = "X_30"
rotation = "initial_state"
file_xray_reference = rotation+"/XrayX_raytracing.0000.bin"
reference_xray = read_binary.get_data_xray(file_xray_reference)
reference_xray = np.transpose(reference_xray, [1,0])

# path = "k_zero/"
path = "initial_state/"
# path = "X_30/"
# path = "X_60/"
# path = "X_90/"
# path = "Y_60/"
# path = "Y_90/"
# path = "Y_30_X_60_n/"
# l = os.listdir(path)

vmin = np.min(reference_xray)+0.1
vmax = np.max(reference_xray)
cmap = cm.get_cmap("inferno")
cmap  = cm.inferno
cmap.set_bad("black")

# fig = plt.figure(figsize=(6.1,5))

ims = []
for i in range(1, 220):
	data_name = "XrayX_raytracing."+str(i).zfill(4)+".bin"
	# folder_index = l.index(data_name)
	print(path+data_name)
	data_xray = read_binary.get_data_xray(path+data_name)
	data_xray = np.transpose(data_xray, [1,0])
	data_xray[data_xray < 1e10] = vmin
	im = plt.imshow(data_xray,
			cmap=cmap, 
			extent=[0,x_size/AU, 0,y_size/AU], 
			origin="lower",
			# interpolation="Nearest",
			norm=LogNorm(vmin=1e24,vmax=1e28))
	plt.xlabel("x[AU]")
	plt.ylabel("y[AU]")
	plt.title("X-ray animation")
	# clb.ax.set_title()
	
	ims.append([im])
clb = fig.colorbar(im, extend="both")
# clb.set_label('label', labelpad=-40, y=1.05, rotation=0)
clb.ax.set_title('$Erg/s$')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=False)
ani.save(path[:-1]+'.mp4', writer=writer)
plt.show()
