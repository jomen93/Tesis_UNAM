import paramiko
from Globals import *
from tqdm import tqdm

def ssh_connect(host, port, username, password):
	"""
	Connect to a remote server using SSH protocol.
	Parameters
	----------
	host: str
		The hostname or IP address of the server.
	port: int 
		The port number to use for the connection.
	username: str
		The username to use for authentication.
	password: str
		The password to use for authentication.

	Returns
	-------
	paramiko.SSHClient: The SSH client  object
		client to perform the conection to remote server 
	"""
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
	ssh.connect(host, port, username=user, password=password) 
	print("Diable conection done!")
	return ssh 

def Progress(transferred, ToBeTransferred):
	"""
	A utility function to display the progress of a download file. The function
	uses tqdm package to display the progress bar
	
	Parameters
	----------
	transferred: int
		The number of bytes that have been transferred.
	ToBeTransferred: int
		The total number of bytes that need to be Transferred.
	
	Returns
	-------
	None
		Auxiliary function to show the download progress
	"""
	with tqdm(total=ToBeTransferred, unit="B", unit_scale=True, unit_divisor=1024, leave=False) as pbar:
		pbar.update(transferred)
    

def download_files(ssh, remote_path, local_path, file_extension):
	"""
	This function download files from a remote server using ssh connection
	
	Parameters
	----------
	ssh: Object
		A valid ssh connection
	remote_path: str
		The remote path where the files are located
	local_path: str
		The local path where files will be saved
	file_extension: str
		The extension of the files to be downloaded
	"""
	cmd = "ls "+data_path+file_extension
	stdout = ssh.exec_command(cmd)[1]
	filelist = stdout.read().splitlines()
	ftp = ssh.open_sftp()
	for file in filelist:
		print("Downloading "+file.decode("utf-8")+" ...")
		dfile = file.decode("utf-8").replace(data_path, local_path)
		ftp.get(file, dfile, callback=Progress)

	ftp.close()


# data_path = "raytracing/data/"
data_path = "/storage2/jsmendezh/raytracing/data/"

# file_extension = "*Cut*.bin"
# file_extension = "parameter.dat"
file_extension = "*XrayX*.bin"

local_path = "/Users/johan/Documents/Tesis/"#+file

ssh = ssh_connect(host, port, user, password)
download_files(ssh, data_path, local_path, file_extension)
ssh.close()