import read_binary
from Globals import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import cm


# Set latex configuration
mpl.rc("text", usetex=True)
mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

# rotation = "state_AXIS_X_10"
# rotation = "initial_state/"
# rotation = "Y_30_X_60_n/"
# rotation = ""
# rotation = "X_30/"
# rotation = "X_60/"
# rotation = "X_90/"
#rotation = "k_0/"
#rotation = "k_25/"
#rotation = "k_250/"
#rotation = "k_1000/"
rotation = "k_2500/"
n = 88
# file_xray = rotation+"/XrayX_raytracing."+str(n).zfill(4)+".bin"
# file_xray_reference = rotation+"/XrayX_raytracing.0001.bin"
file_xray = rotation+"XrayX_raytracing."+str(n).zfill(4)+".bin"
file_xray_reference = rotation+"XrayX_raytracing.0001.bin"
# file_xray = rotation+"XrayX."+str(n).zfill(4)+".bin"
# file_xray_reference = rotation+"XrayX.0001.bin"

data_xray = read_binary.get_data_xray(file_xray)
# data_xray = np.transpose(data_xray, [1,0])

reference_xray = read_binary.get_data_xray(file_xray_reference)
vmin = np.min(reference_xray)+0.1
vmax = np.max(reference_xray)
cmap = cm.get_cmap("inferno")
# data_xray[data_xray < 1e10] = vmin
name = "X_ray_"+str(n).zfill(4)
fileout = name+".png"
plt.figure()
cmap  = cm.inferno
cmap.set_bad("black")
im = plt.imshow(data_xray, 
				cmap=cmap, 
				# extent=[0,x_size/AU, 0,y_size/AU], 
				origin="lower",
				interpolation="Nearest",
				# norm=LogNorm(vmin=1e-10,vmax=1e2),
				norm=LogNorm(vmin=1e25,vmax=1e28),
				#vmax=max_ref, 
				#vmin=min_ref			
				)
plt.xlabel("x[AU]")
plt.ylabel("y[AU]")
plt.title("Xray map")
plt.colorbar(im, extend="both")
plt.savefig(fileout, transparent=True)
plt.show()

print(data_xray.sum())

###


# Paso 1. utilizar el coldens para hacer la proyeccion de la densidad y comparar las
# regiones de emisión con el mapa de rayos x que se tiene 

# Revisión profundidad óptica

# Paso 2. Hacer otra simulación con parámetros mas violentos, para ver si 
# se tiene una mayor variación en la curva de rayos x


# The spherical wind parameters of wind WC7

# wind1%xc = x1 
# wind1%yc = y1 
# wind1%zc = z1
# wind1%radius = 2.0 * AU
# wind1%mdot = 1e-5 * MSUN/YR
# wind1%vinf = 2000 * KPS
# wind1%temp = 1.0e5
# wind1%mu = mui

# ! The spherical wind parameters of wind O4-5
# wind2%xc = x2 
# wind2%yc = y2 
# wind2%zc = z2
# wind2%radius = 2.0 * AU
# wind2%mdot = 1e-6 * MSUN/YR
# wind2%vinf = 1000 * KPS
# wind2%temp = 1.0e4
# wind2%mu = mui

