"""This script allows to automate the fitting of saturated images."""


import os
import numpy as np
from astropy.io import fits
from functions import *

def automation(table,files,target):

	sources = open_fits_table(table)
	mu,theta,FWHM = get_parameters(sources)

	for image in files:
		grid,data = open_file(image)
		X,Y = grid
		xl,xr,yb,yt = X[0,0],X[0,-1],Y[0,0],Y[0,-1]

	


		
