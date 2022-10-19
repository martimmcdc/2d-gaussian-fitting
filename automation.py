#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""This script allows to automate the fitting of saturated images."""


import os
import numpy as np
from astropy.io import fits
from matplotlib.colors import Normalize,LogNorm
from functions import *
from visual_methods import *

def automation(table,files,path='',save_path='',wavelength='',show=False,**kwargs):

	sources = open_fits_table(path+table)
	mu,theta,FWHM = get_parameters(sources,wavelength=wavelength)
	new_list = []

	for image in files:
		data,grid = open_fits_image(path+image)
		header = fits.getheader(path+image)
		X,Y = grid
		xl,xr,yb,yt = grid_lims(grid)
		in_window = (mu[:,0]<=xl)&(mu[:,0]>=xr)&(mu[:,1]>=yb)&(mu[:,1]<=yt)

		mu0,theta0,FWHM0 = mu[in_window],theta[in_window],FWHM[in_window]
		peaks = len(mu0)
		data_fit,params,bg = fitter(data,grid,peaks,mu0,theta0,FWHM0,**kwargs)

		header_fit = header.copy()
		header_fit['COMMENT'] = 'Image fixed using {n} gaussian sources'.format(n=peaks)
		fits.writeto(save_path+image[:-5]+'_fix.fits',data_fit,header_fit,overwrite=True)
		new_list.append(image[:-5]+'_fix.fits')

		if show==True:
			vmin,vmax = data_fit.min(),data_fit.max()
			lin_norm = Normalize(vmin,vmax)
			log_norm = LogNorm(vmin,vmax)
            
			fig,axs = plt.subplots(figsize=(9,9))
			axs.set_title('Fixed image for file '+image,fontsize=14)
			im1 = axs.imshow(data_fit,origin='lower',extent=[xl,xr,yb,yt],norm=log_norm)  
			axs.scatter(mu0[:,0],mu0[:,1],color='black',marker='+')
			axs.scatter(params[::6],params[1::6],color='red',marker='+')
			plt.colorbar(im1)
			plt.show()
	return new_list