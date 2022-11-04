#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""This script allows to automate the fitting of saturated images."""


import os
import numpy as np
from astropy.io import fits
from matplotlib.colors import Normalize,LogNorm
import dfitspy
from functions import *
from visual_methods import *

def automation(table,files,path='',save_path='',wavelength='',show=False,**kwargs):

	sources = open_fits_table(path+table)
	mu,theta,FWHM = get_parameters(sources,wavelength=wavelength)
    
	new_list = []
	if files == ["all"]:
		all_files = dfitspy.get_files(["all"], path[:-1])
		files_sort = dfitspy.dfitsort(all_files,["all"],exact=True)
		files = list(dict(files_sort.items()))
		files.remove(table)

	for image in files:
		print(image)
		data,grid = open_fits_image(path+image)
		header = fits.getheader(path+image)
		X,Y = grid
		xl,xr,yb,yt = grid_lims(grid)
		in_window = (mu[:,0]<=xl)&(mu[:,0]>=xr)&(mu[:,1]>=yb)&(mu[:,1]<=yt)

		mu0,theta0,FWHM0 = mu[in_window],theta[in_window],FWHM[in_window]
		peaks = len(mu0)
        
		x = header['crval1'] + header['cdelt1'] * (np.arange(0,header['naxis1'],1) - header['crpix1'])
		y = header['crval2'] + header['cdelt2'] * (np.arange(0,header['naxis2'],1) - header['crpix2'])
		xl,xr,yb,yt = x.max(),x.min(),y.min(),y.max()
		plt.figure(figsize=(8,8))
		plt.imshow(np.log10(data),origin='lower',extent=(xl,xr,yb,yt))
		plt.scatter(mu0[:,0],mu0[:,1],color='black',marker='+')
		plt.xlabel(header['ctype1']+' [{}]'.format(header['cunit1']))
		plt.ylabel(header['ctype2']+' [{}]'.format(header['cunit2']))
		plt.show()
        
		data_fit,params,bg = fitter(data,grid,peaks,mu0,theta0,FWHM0,**kwargs)
        
		gaussians = gaussianMult(grid,*params) + bg
		resid = residuals(grid,data,params,bg)
        
		header_fit = header.copy()
		header_fit['COMMENT'] = 'Image fixed using {n} gaussian sources'.format(n=peaks)
		fits.writeto(save_path+image[:-5]+'_fix.fits',data_fit,header_fit,overwrite=True)
		fits.writeto(save_path+image[:-5]+'_res.fits',resid,header_fit,overwrite=True)
		new_list.append(image[:-5]+'_fix.fits')        
        
		if show==True:
			xl,xr,yb,yt = grid_lims(grid)

			vmin,vmax = data_fit.min(),data_fit.max()
			lin_norm = Normalize(vmin,vmax)
			log_norm = LogNorm(vmin,vmax)
    
			fig,[ax0,ax1] = plt.subplots(1,2,figsize=(10,5))

			im0 = ax0.imshow(data_fit,origin='lower',extent=[xl,xr,yb,yt],norm=log_norm)
			ax0.set_title('Fitted image (log scale)')
			ax0.scatter(mu0[:,0],mu0[:,1],color='black',marker='+')
			ax0.scatter(params[::6],params[1::6],color='red',marker='+')

			im1 = ax1.imshow(resid,origin='lower',extent=[xl,xr,yb,yt])
			ax1.set_title('Residuals image')
			ax1.scatter(mu0[:,0],mu0[:,1],color='red',marker='+')

			plt.colorbar(im0,ax=ax0,shrink=0.75)
			plt.colorbar(im1,ax=ax1,shrink=0.75)

	fig.tight_layout()
	plt.show()

	return new_list