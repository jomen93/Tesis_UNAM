import numpy as np 
import matplotlib.pyplot as plt
from Globals import *

def get_data(file):
	"""
	Read an process data from a given binary file
	Parameters
	----------
	file: str
		Name of the file to read
	Returns
	-------
	numpy.ndarray
		The read and processed data in a Numpy matrix
	"""
	data = np.memmap(file, dtype="d", mode="r", shape=(neq, nx, ny))
	data = np.transpose(data, [0,2,1])
	return data


def get_data_xray(file):
    """
    Read and process data from a given file.
    
    Parameters
    ----------
    file : str
        Name of the file to read.
    
    Returns
    -------
    numpy.ndarray
        The read and processed data in a NumPy matrix.
    
    """
    data = np.fromfile(file, dtype="d", count=nx*ny)
    return data.reshape((nx, ny), order="F")


def reference_value(file_reference, plot_usage):
	"""This function is used to calculate the minimum and maximum
	values of a reference data set. It takes two parameters, the
	first one is the name of the reference file, the second one is
	an integer that indicate which data set is going to be used

	Parameters
	----------
	file_reference : str
		Name of the reference file.
	plot_usage : int
		Indicator of which data set is going to be used

	Returns
	-------
	Tuple
		A tuple containing the minimum and maximum values of the 
		selected data set
	"""
	data = get_data(file_reference)
	data = np.transpose(data, [0, 2, 1])
	return np.min(data[plot_usage]), np.max(data[plot_usage])

