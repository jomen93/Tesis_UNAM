import os 
import paramiko
from Globals import *


# Make the Client
ssh = paramiko.SSHClient()
# to avoid "not found in known_hosts"
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
# Make the coenction 
ssh.connect(host, port, username=user, password=password) 
print("Diable conection done! ")

# list the files to download, we list only bimn format
# data_path = "raytracing/data/"
data_path = "/storage2/jsmendezh/raytracing/data/"
# bins = "*Cut*.bin"
# bins = "parameter.dat"
bins = "*XrayX*.bin"
# bins = "*XrayX.*.bin"
# Execute the the command through ssh 
# cmd = "ls "+data_path+bins
cmd = "ls "+data_path+bins
stdin, stdout, stderr = ssh.exec_command(cmd)
filelist = stdout.read().splitlines()

# Auxiliary function to print progress
def Progress(transferred, ToBeTransferred):

	print("transferred: {:.1f}% \t Out of 100 %".format((transferred/ToBeTransferred)*100), end="\r")

# # Construction of the data transfer protocol
ftp = ssh.open_sftp()
local_path = "/Users/johan/Documents/Tesis/"#+file


for file in filelist:
	print("Downloading "+file.decode("utf-8")+" ...")
	dfile = file.decode("utf-8").replace(data_path, local_path)
	ftp.get(file, dfile, callback=Progress)

ftp.close()
ssh.close()





