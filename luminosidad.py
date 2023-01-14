import numpy as np 
import read_binary
from Globals import *
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc("text", usetex=True)
mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

data_len = 150
dtout = 0.1 * YR
tfin  = 2 * 10.98 * YR
nout = int(tfin/dtout)
time = np.linspace(0,tfin,nout)
Pe = 10.98 * YR
phase = list()
xray_dots = list()

for i in range(data_len):
	file_xray = "k_0/XrayX_raytracing."+str(i).zfill(4)+".bin"
	data_xray = read_binary.get_data_xray(file_xray)
	phase_step = ((time[i]*t_sc)%Pe)/Pe +0.25
	xray_dots.append(data_xray.sum())
	phase.append(phase_step)

xray_dots = np.array(xray_dots[1:])
phase = np.array(phase[1:])

plt.plot(phase, xray_dots, "r.", markersize=3.0)
plt.grid(True)
plt.xlabel("$\\phi$")
plt.ylabel("$L_{x}$")
plt.ylim([np.min(xray_dots)*0.8, np.max(xray_dots)*1.1])
plt.show()