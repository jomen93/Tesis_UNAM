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

# path to data
xpath_k_100 = "k/"
xpath_k_0 = "k_0/"
xpath_k_25 = "k_25/"
xpath_k_250 = "k_250/"
xpath_k_1000 = "k_1000/"
xpath_k_2500 = "k_2500/"
xpath = "initial_state/"
xpath_30 = "X_30/"
xpath_60 = "X_60/"
xpath_90 = "X_90/"
ypath_30 = "Y_30/"
# xpath = "Y_30_X_60_n/"

# files = [xpath_k, xpath, xpath_30, xpath_60, xpath_90, xpath_k_100]
# names_files = ["Opacidad = 0", "0", "30", "60", "90", "Opacidad = 100"]

files = [xpath_k_0, xpath_k_25, xpath_k_250, xpath_k_1000, xpath_k_2500]
names_files = ["0", "25", "250", "1000","2500"]

# cycle to read all binary files in the folder 
n = 219
# list to save xray emission for each file 
p_xray = []
# list to sve data of each emission
xray = []
# get the map of figure in imshow
cmap = cm.get_cmap("inferno")
# set the orbital period in the system 
Pe = 10.98 * YR
dtout = 0.1 * YR
tfin = 2 * 10.98 * YR
nout = int(tfin/dtout)

time = np.linspace(0, tfin, nout+1)
theta = (time+0.25*Pe)/Pe
# Definition matrix of angles
p_xray = np.zeros((len(files), n+1))

# cycle to run over data
for j in range(len(files)):
	for i in range(n+1):

		file_xray = files[j]+"/XrayX_raytracing."+str(i).zfill(4)+".bin"
		
		data_xray = read_binary.get_data_xray(file_xray)

		# data_xray.sum()
		p_xray[j][i] = data_xray.sum()

# # make a multiplot
# fig, axis = plt.subplots(2, 2, figsize=(15,10))


# # Initial state
# axis[0, 0].plot(theta[1:], p_xray[1][1:], "b.", label="$30^{o}$")
# axis[0, 0].plot(theta[1:], p_xray[2][1:], "g.", label="$60^{o}$")
# axis[0, 0].plot(theta[1:], p_xray[3][1:], "k.", label="$90^{o}$")
# axis[0, 0].plot(theta[1:], p_xray[0][1:], "r.", label="$0^{o}$")
# axis[0, 0].legend()
# axis[0, 0].grid(True)
# axis[0, 0].set_xlabel("Phase")
# axis[0, 0].set_ylabel("Xray [$erg/s$]")

# # 30 degres about X-axis
# axis[0, 1].plot(theta[1:], p_xray[1][1:], "r-")
# axis[0, 1].plot(theta[1:], p_xray[1][1:], "r.")
# # 60 degres about X-axis
# axis[1, 0].plot(theta[1:], p_xray[2][1:], "r-")
# axis[1, 0].plot(theta[1:], p_xray[2][1:], "r.")
# # 90 degres about X-axis
# axis[1, 1].plot(theta[1:], p_xray[3][1:], "r-")
# axis[1, 1].plot(theta[1:], p_xray[3][1:], "r.")

# # phase = mod(time*t_sc,Pe)/Pe +0.25
# plt.show()

x_common = theta[2:]

fig = plt.figure()
plt.plot(x_common, p_xray[0][2:], "k-", label="$\\kappa = "+names_files[0]+"$", linewidth=0.6)
plt.plot(x_common, p_xray[1][2:], "b-", label="$\\kappa = "+names_files[1]+"$", linewidth=0.6)
plt.plot(x_common, p_xray[2][2:], "g-", label="$\\kappa = "+names_files[2]+"$", linewidth=0.6)
plt.plot(x_common, p_xray[3][2:], "m-", label="$\\kappa = "+names_files[3]+"$", linewidth=0.6)
plt.plot(x_common, p_xray[4][2:], "r-", label="$\\kappa = "+names_files[4]+"$", linewidth=0.6)
#plt.plot(x_common[69:78], p_xray[0][71:80], "ko")
# plt.plot(x_common, p_xray[4][2:], "r-", label="$"+names_files[3]+"^{o}$", linewidth=0.6)
# plt.plot(x_common, p_xray[5][2:], "m-", label="$"+names_files[5]+"^{o}$", linewidth=0.6)
plt.xlabel("$\phi$")
plt.ylabel("xray$[Erg/s]$")
plt.title("x-ray curve")
plt.legend()
plt.grid(b=True, which="major", linestyle='-')
plt.grid(b=True, which="minor", linestyle='--', alpha=0.3)
plt.minorticks_on()
plt.savefig("first_xray")
plt.show()

dif = np.mean(p_xray[0] - p_xray[3])
print(dif)

# plt.plot(0, p_xray[0].sum(), "b.", label="$"+names_files[0]+"^{o}$ X")
# plt.plot(30, p_xray[1].sum(), "g.", label="$"+names_files[1]+"^{o}$ X")
# plt.plot(60, p_xray[2].sum(), "k.", label="$"+names_files[2]+"^{o}$ X")
# plt.plot(90, p_xray[3].sum(), "r." , label="$"+names_files[3]+"^{o}$ X")
# plt.plot(30, p_xray[4].sum(), "y." , label="$"+names_files[4]+"^{o}$ Y")
# plt.grid(True)
# plt.legend()
# plt.show()

