"""This script allows to automate the fitting of saturated images."""


import os
import numpy as np
from astropy.io import fits
from functions import *

def automation(table,files,target,path='',wavelength='',**kwargs):

	sources = open_fits_table(path+table)
	mu,theta,FWHM = get_parameters(sources,wavelength=wavelength)

	for image in files:
		data,grid = open_fits_image(path+image)
		X,Y = grid
		xl,xr,yb,yt = grid_lims(grid)
		in_window = (mu[:,0]<=xl)&(mu[:,0]>=xr)&(mu[:,1]>=yb)&(mu[:,1]<=yt)

		mu0,theta0,FWHM0 = mu[in_window],theta[in_window],FWHM[in_window]
		peaks = len(mu0)
		data_fit,params,bg = fitter(data,grid,peaks,mu0,theta0,FWHM0,**kwargs)

	


		
