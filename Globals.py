parameters_file = "parameter.dat"
credentials_file = "credentials.dat"

with open(parameters_file) as pf:
	parameters = pf.readlines()

with open(credentials_file) as cf:
	credentials = cf.readlines()


# physical parameters
x_size 		= float(parameters[0].split()[1])
y_size 		= float(parameters[1].split()[1])
z_size 		= float(parameters[2].split()[1])
nx  		= int(parameters[3].split()[1])
ny          = int(parameters[4].split()[1])
nz          = int(parameters[5].split()[1])
max_res_x   = int(parameters[6].split()[1])
max_res_y   = int(parameters[7].split()[1])
max_res_z   = int(parameters[8].split()[1])
max_lev_res = int(parameters[9].split()[1])
neq 		= int(parameters[16].split()[1]) 
CFL 		= float(parameters[20].split()[1]) 
eta		 	= float(parameters[21].split()[1]) 
gamma 		= float(parameters[22].split()[1]) 
mu0 		= float(parameters[23].split()[1]) 
mui 		= float(parameters[24].split()[1]) 
t_sc 		= float(parameters[-1].split()[1]) 

AMU  = 1.660538782e-24
KB   = 1.380650400e-16
PC   = 3.085677588e+18
AU   = 1.495978707e+13
YR   = 3.155673600e+7
KYR  = 3.155673600e+10
MSUN = 1.988920000e+33
KPS  = 1.0e5
PI   = 3.14159265358979

# server parameters
host	 = credentials[0].split()[1]
port 	 = credentials[1].split()[1]
user 	 = credentials[2].split()[1]
password = credentials[3].split()[1]