import numpy as np 
import matplotlib.pyplot as plt 
from Globals import *
import read_binary
import matplotlib as mpl
from Sistema_Binario import BinarySystem

# Set latex configuration
mpl.rc("text", usetex=True)
mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

M1 = 31 * MSUN
M2 = 12 * MSUN
Pe = 10.98 * YR
ecc = 0.61
Rx = 30 * AU
Ry = 30 * AU
pos = 8

binary = BinarySystem(M1, M2, Pe, ecc)
x1, y1, x2, y2 = binary.trajectories()

plt.plot(x1, y1, "ro", markersize=1)
plt.plot(x2, y2, "bo", markersize=1)
plt.grid(True)
plt.axis("square")
plt.xlim([0, 60])
plt.ylim([0, 60])
plt.xlabel("x[AU]")
plt.xlabel("y[AU]")
plt.title(f"Trayectorias t = {binary.time[pos]/YR:.2f} años")
plt.show()






# dist = np.sqrt((pos1x[pos] - pos2x[pos])**2 + (pos1y[pos] - pos2y[pos])**2)
# print(f"Distancia entre estrellas = {dist:.2f} en el tiempo = {pos}")
# plt.plot([pos1x[pos], pos2x[pos]], [pos1y[pos], pos2y[pos]], "k--", label=f"d = {dist:.2f} AU")

# file = "cuts/CutZ."+str(pos).zfill(4)+".bin"
# # file = "CutZ."+str(pos).zfill(4)+".bin"
# data = read_binary.get_data(file)
# rho = data[0]
# r1 = [pos1y[pos], pos1x[pos]]
# r2 = [pos2y[pos], pos2x[pos]]
# print(r1, r2)
# rdir = [(r2[0]-r1[0])/dist, (r2[1]-r1[1])/dist] #####
# r1[0] += 0.4*rdir[0]*dist
# r1[1] += 0.4*rdir[1]*dist 
# r = r1  
# rnorm = 0
# ds = 60/256
# d = list()
# rho_line = list()
# while rnorm < dist*0.6:

# 	r[0] += ds * rdir[0]
# 	r[1] += ds * rdir[1]
# 	#print(f"x = {r[0]}, y = {r[1]}")
# 	rnorm += ds
# 	d.append(rnorm)
# 	xi = int(r[0]/60 * nx)
# 	yi = int(r[1]/60 * ny)
# 	#print(f"xint = {xi},yint = {yi}")
# 	print(rho[xi,yi])
# 	rho_line.append(rho[xi,yi])

# ind_rhomax = np.argmax(rho_line)
# rho_bef = np.array(rho_line[ind_rhomax -10 : ind_rhomax-7])
# rho_aft = np.array(rho_line[ind_rhomax+1 : ind_rhomax+4])
# compresibility = np.mean(rho_aft)/np.mean(rho_bef)
# print(f"valor antes del choque = {np.mean(rho_bef)}")
# print(f"valor después del choque = {np.mean(rho_aft)}")
# print(f"valor de la compresibilidad = {compresibility}")
# plt.plot(d, rho_line, "k-", label=f"d = {dist:.2f} AU")
# plt.plot(d[ind_rhomax], rho_line[ind_rhomax], "ro")
# plt.plot(d[ind_rhomax -10: ind_rhomax-7], rho_line[ind_rhomax -10 : ind_rhomax-7], "bo")
# plt.plot(d[ind_rhomax +1: ind_rhomax+4], rho_line[ind_rhomax +1 : ind_rhomax+4], "bo")
# plt.grid(True)
# plt.xlabel("Distancia entre estrellas [AU]")
# plt.ylabel("Densidad [$g/cm^{3}$]")
# plt.title("Distribución densidad")
# plt.legend()
# plt.savefig("Compresibilidad")
# plt.show()

# print(f"Distancia entre estrellas = {dist:.2f} en el tiempo = {pos}")
# plt.plot([pos1x[pos], pos2x[pos]], [pos1y[pos], pos2y[pos]], "k--", label=f"d = {dist:.2f} AU")
# plt.plot(pos1x, pos1y, "ro", markersize=1)
# plt.plot(pos2x, pos2y, "bo", markersize=1)
# plt.grid(True)
# plt.axis("square")
# plt.xlim([0, 60])
# plt.ylim([0, 60])
# plt.xlabel("x[AU]")
# plt.xlabel("y[AU]")
# plt.title(f"Trayectorias t = {time[pos]/YR:.2f} años")
# plt.legend()
# # plt.savefig("trayectorias")
# plt.show()




