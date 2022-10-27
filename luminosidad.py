import numpy as np 
import read_binary
from Globals import *

import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rc("text", usetex=True)
mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

# Elección de rotacion
hard= ""
soft= "initial_state/"

root =[hard, soft]
data_len = 150
# Definicion del tiempo de simulacion 
dtout = 0.1 * YR
tfin  = 2 * 10.98 * YR
nout = int(tfin/dtout)
time = np.linspace(0,tfin,nout)
Pe = 10.98 * YR

# lista para guardar cada caso 
case = list()
phase_tot = list()

for cases in root:
	
	phase = list()
	xray_dots = list()


	for i in range(data_len):

		# Archivo de lectura
		file_xray = cases+"XrayX_raytracing."+str(i).zfill(4)+".bin"
		# lectura de datos 
		data_xray = read_binary.get_data_xray(file_xray)
		# calculo de la phase 
		phase_step = ((time[i]*t_sc)%Pe)/Pe +0.25
		phase.append(phase_step)

		xray_dots.append(data_xray.sum())

	case.append(xray_dots)
	phase_tot.append(phase)

phase_tot = np.array(phase_tot)
case = np.array(case)

plt.plot(case[0], "r-", label="Caso fuerte")
plt.plot(case[1], "b-", label="Caso débil")
plt.grid(True)
plt.xlabel("$\\phi$")
plt.ylabel("$L_{x}$")
plt.ylim([1e30, 4e32])
plt.legend()
plt.show()