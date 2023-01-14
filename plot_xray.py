#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Johan Méndez
Created on: 2022-02-15
Description: File to calculate compresibility limit in the binary 
start systems 
"""
import read_binary
from Globals import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import cm

# Set latex configuration
mpl.rc("text", usetex=True)
mpl.rc("font", **{"family":"serif", "serif":["Computer Modern"]})

rotation_options = {
	"X_30":"X_30",
	"X_60":"X_60",
	"X_90":"X_90",
	"k_0":"k_0",
	"k_25":"k_25",
	"k_250":"k_250",
	"k_1000":"k_1000",
	"k_2500":"k_2500",
}

def plot_xray(n, rotation, save_image = False):
	"""
	This function is used to plot a X-ray map from a binary file.
	It uses the matplotlib library to create the plot and the 
	get_data_xray function to read the binary file

	Parameters
	----------
	n: int
		The number of the X-ray file.
	rotation: str
		The rotation of the image, it could be a dictionary name
	save_image: bool, optional
		A flag to indicate wheter the image should be saved or not 
		(Default is False)
	
	Returns
	-------
	None
		The function creates and displays the image should be saves the 
		image if save_image is set to True
	"""
	file_xray = f"{rotation}/XrayX_raytracing.{str(n).zfill(4)}.bin"
	data_xray = read_binary.get_data_xray(file_xray)
	cmap = cm.get_cmap("inferno").copy()
	cmap.set_bad("black")
	name = f"X_ray_{str(n).zfill(4)}"
	fileout = f"{name}.png"
	plt.figure()
	im = plt.imshow(data_xray, 
					cmap=cmap, 
					extent=[0,x_size/AU, 0,y_size/AU], 
					origin="lower",
					interpolation="Nearest",
					norm=LogNorm(vmin=1e25,vmax=1e28),	
					)
	plt.xlabel("x[AU]")
	plt.ylabel("y[AU]")
	xray_total = data_xray.sum()
	plt.title(f"Xray map = {xray_total:.2e}")
	plt.colorbar(im, extend="both")
	if save_image == True:
		plt.savefig(fileout, transparent=True)
	plt.show()

n = 15
rotation = rotation_options["k_0"]

plot_xray(n, rotation)

