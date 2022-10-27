import numpy as np 
import matplotlib.pyplot as plt
from Globals import *

def get_data(file):
	data = np.fromfile(file, dtype="d", count=neq*nx*ny)
	data = data.reshape((neq,nx,ny), order="F")
	data = np.transpose(data, [0, 2, 1])
	return data

def get_data_xray(file):
	data = np.fromfile(file, dtype="d", count=nx*ny)
	return data.reshape((nx, ny), order="F")

def reference_value(file_reference, plot_usage):
	data = get_data(file_reference)
	data = np.transpose(data, [0, 2, 1])
	return np.min(data[plot_usage]), np.max(data[plot_usage])

