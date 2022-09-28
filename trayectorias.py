import numpy as np 
import matplotlib.pyplot as plt 
from Globals import *
import read_binary
import matplotlib as mpl

# Set latex configuration
mpl.rc("text", usetex=True)
mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

M1 = 31 * MSUN
M2 = 12 * MSUN
Pe = 10.98 * YR
ecc = 0.61
Rx = 30 * AU
Ry = 30 * AU
GRAV = 6.67259e-8

def ComputeBinary(phase):

	tol = 1e-10

	mu = (M1*M2)/(M1+M2)
	a = (Pe*Pe*GRAV*(M1+M2)/(4*PI*PI))**(1.0/3.0)
	L = mu*(1-ecc)*a*np.sqrt((1+ecc)/(1-ecc)*GRAV*(M1+M2)/a)

	# Anomalía media
	M = 2*PI*phase

	if M == 0:
		E = 0

	if ecc > 0.8:
		x = np.copysign(PI, M)
	else:
		x = M

	rel_err = 2*tol

	while rel_err > tol:
		xn = x - (x-ecc*np.sin(x)-M)/(1-ecc*np.cos(x))
		if x != 0:
			rel_err = abs((xn-x)/x)
			x = xn

	E = x
	theta = 2.0*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(E/2))
	rad = a*(1.0-ecc*ecc)/(1+ecc*np.cos(theta))

	thetadot = L/(mu*rad**2)
	raddot = a*ecc*np.sin(theta)*(1-ecc**2)/(1+ecc*np.cos(theta))**2 * thetadot

	radx = rad*np.cos(theta)
	rady = rad*np.sin(theta)
	x1 = Rx - M2/(M1+M2)*radx
	y1 = Ry - M2/(M1+M2)*rady
	x2 = Rx + M1/(M1+M2)*radx
	y2 = Ry + M1/(M1+M2)*rady

	vx = raddot*np.cos(theta)-rad*thetadot*np.sin(theta)
	vy = raddot*np.sin(theta)+rad*thetadot*np.cos(theta)
	vx1 = -M2/(M1+M2)*vx
	vy1 = -M2/(M1+M2)*vy
	vx2 = M1/(M1+M2)*vx
	vy2 = M1/(M1+M2)*vy

	return x1, y1, x2, y2, vx1, vy1, vx2, vy2

dtout = 0.1 * YR
tfin = 2 * 10.98 * YR

nout = int(tfin/dtout)

time = np.linspace(0, tfin, nout)

# ciclo para obtener las posiciones
pos1x = []
pos1y = []
pos2x = []
pos2y = []
vel1 = []
vel2 = []

for i in range(len(time)):
	phase = (time[i]/Pe + 0.25)% 1
	#print(f"Fase = {phase:.2f}")
	r = ComputeBinary(phase)
	pos1x.append(r[0])
	pos1y.append(r[1])
	pos2x.append(r[2])
	pos2y.append(r[3])

# Ciclo para obtener las distancias entre estrellas de todos los pasos de tiempo 
dist_step = []
for i in range(len(pos1x)): 
	dist_step.append(np.sqrt((pos1x[i] - pos2x[i])**2 + (pos1y[i] - pos2y[i])**2)/AU)


pos = 7

# file = "cuts/CutZ."+str(pos).zfill(4)+".bin"
file = "Cuts/CutZ."+str(pos).zfill(4)+".bin"
data = read_binary.get_data(file)
rho = data[0]
dist = np.sqrt((pos1x[pos] - pos2x[pos])**2 + (pos1y[pos] - pos2y[pos])**2)
r1 = [pos1y[pos], pos1x[pos]]
r2 = [pos2y[pos], pos2x[pos]]
#print(r1, r2)
rdir = [(r2[0]-r1[0])/dist, (r2[1]-r1[1])/dist] 

r = r1  
rnorm = 0
ds = 0.01 * AU
# Se supone que el corte es en el eje z
nx = data.shape[1]
dx = x_size/nx
ny = data.shape[2]
dy = y_size/ny
d = list()
rho_line = list()
while rnorm < dist:

	r[0] += ds * rdir[0]
	r[1] += ds * rdir[1]
	rnorm += ds
	d.append(rnorm)
	xi = int(r[0]/dx)
	yi = int(r[1]/dy)
	#print(f"xint = {xi},yint = {yi}")
	#print(rho[xi,yi])
	rho_line.append(rho[xi,yi])

rho_line = np.array(rho_line)

rho_line_forward = np.roll(rho_line, -1)

mask_monotony_left = (rho_line >= rho_line_forward)
mask_monotony_left = np.where(mask_monotony_left[1:-1] == False)[0]

#for i in range(len(mask_monotony)):
#	print(mask_monotony[i])

plt.plot(d, rho_line, "k-", label=f"d = {dist/AU:.2f} AU")
plt.plot(d[mask_monotony_left[0]], rho_line[mask_monotony_left[0]], "bo")
plt.plot(d[mask_monotony_left[3]], rho_line[mask_monotony_left[3]], "bo")

print(rho_line[mask_monotony_left[3]]/rho_line[mask_monotony_left[0]])
plt.semilogy()
plt.grid(True)
plt.xlabel("Distancia entre estrellas [AU]")
plt.ylabel("Densidad [$g/cm^{3}$]")
plt.title("Distribución densidad")
plt.legend()
plt.savefig("Compresibilidad", transparent=True)
plt.show()


plt.plot([pos1x[pos]/AU, pos2x[pos]/AU], [pos1y[pos]/AU, pos2y[pos]/AU], "k--", label=f"d = {dist/AU:,.1f} AU")
plt.plot([pos1x[pos]/AU], [pos1y[pos]/AU], "k.")
plt.plot([pos2x[pos]/AU], [pos2y[pos]/AU], "k.")
plt.plot(np.array(pos1x)/AU, np.array(pos1y)/AU, "b-",label="Trayectoria 1")
plt.plot(np.array(pos2x)/AU, np.array(pos2y)/AU, "r", label ="Trayectoria 2")
print(f"Distancia entre estrellas = {dist/AU:.2f} en el tiempo = {pos}")

plt.grid(True)
plt.axis("square")
plt.xlim([0, 50])
plt.ylim([0, 50])
plt.xlabel("x[AU]")
plt.xlabel("y[AU]")
plt.title(f"Trayectorias t = {time[pos]/YR:.2f} años")
plt.legend()
plt.savefig("trayectorias")
plt.show()




