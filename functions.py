"""This file contains mathematical and other functions used in the rest of the program."""


### Imports ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits



### Mathematical functions and models ###

def gaussian(points,mx,my,N,theta,FWHMx,FWHMy):
	"""
	Gaussian function in 2D:
		- points = (x,y) is the grid array at which the function is being evaluated
		- (mx,my) = (mu_x,mu_y) is the centre of the distribution
		- N is an arbitrary normalization constant
		- theta is the angle between the semi-major axis and the horizontal
		- FWHM = (amaj,amin) is given in the same units as the 'points' argument
	"""
	a = 2*np.sqrt(2*np.log(2))
	sigmax,sigmay = FWHMx/a,FWHMy/a
	alphax,alphay = 1/(2*sigmax**2),1/(2*sigmay**2)
	x,y = points
	sin,cos = np.sin(theta),np.cos(theta)
	xl = x*cos - y*sin
	yl = x*sin + y*cos
	mxl = mx*cos - my*sin
	myl = mx*sin + my*cos
	z = N * np.exp( - alphax*(xl-mxl)**2 - alphay*(yl-myl)**2 )
	return z

def gaussianMult(points,*args):
	""" Sum multiple 2D gaussian functions at baseline = 0. """
	z = 0
	for i in range(len(args)//6):
		mx,my,N,theta,FWHMx,FWHMy = args[6*i:6*(i+1)]
		z += gaussian(points,mx,my,N,theta,FWHMx,FWHMy)
	return z

def background(data,method='hist'):
	"""
	Background estimation for astronomical images.
	The method argument selects the method of computation:
		- 'hist':
			Set background as maximum value of intensity histogram;
		- 'mode':
			Set background as mode of the dataset,
			where mode = 2.5*median - 1.5*mean;
		- otherwise background gets set to zero.
	"""
	if method == 'hist':
		yhist,xhist = np.histogram(data,bins=len(np.unique(data)))
		bg = xhist[np.argmax(yhist)]
	elif method == 'mode':
		bg = 2.5*np.median(data) - 1.5*np.mean(data)
	else:
		bg = 0
	return bg

def residuals(grid,data,params,bg):
	sat = np.isnan(data)
	gaussians = gaussianMult(grid,*params) + bg
	residuals = data.copy()
	residuals[~sat] = np.abs(data[~sat]-gaussians[~sat])/data[~sat]
	return residuals



#### File handeling functions ###

def open_fits_image(file,lims=[],show=False):
	"""
	Display a 2D array image from a standard FITS file.
	This function assumes the coordinates to be the galactic system,
	where longitude increases from right to left
	and latitude increases from bottom to top,
	both in degrees.
	The lims argument is a list which, if given, must contain:
	1. Left limit (xl)
	2. Right limit (xr)
	3. Bottom limit (yb)
	4. Top limit (yt)
	of the window in this order.
	"""
    
	# Open and read
	hdulist = fits.open(file)
	hdu = hdulist[0]
	header = hdu.header
	data = hdu.data

	# Get axes right
	x = header['crval1'] + header['cdelt1'] * (np.arange(0,header['naxis1'],1) - header['crpix1'])
	y = header['crval2'] + header['cdelt2'] * (np.arange(0,header['naxis2'],1) - header['crpix2'])
	
	# If window limits are given
	if len(lims)==0:
		xl,xr,yb,yt = x.max(),x.min(),y.min(),y.max()
	else:
		xl,xr,yb,yt = lims
	xsub = x[(x<=xl)&(x>=xr)]
	ysub = y[(y>=yb)&(y<=yt)]
	data_sub = data[(y>=yb)&(y<=yt),:][:,(x<=xl)&(x>=xr)]

	# If image is to be displayed
	if show:
		plt.figure(figsize=(8,8))
		plt.imshow(np.log10(data_sub),origin='lower',extent=(xl,xr,yb,yt))
		plt.xlabel(header['ctype1']+' [{}]'.format(header['cunit1']))
		plt.ylabel(header['ctype2']+' [{}]'.format(header['cunit2']))
		plt.show()

	grid = np.meshgrid(xsub,ysub)
	return grid,data_sub

def open_fits_table(file,ext=1):
	"""Open table in .fits format, which is usually in the 1st extension."""

	# Open and read
	hdulist = fits.open(file)
	hdu = hdulist[ext]
	header = hdu.header
	table = hdu.data

	# Return pandas DataFrame object
	return pd.DataFrame(table)


def get_parameters(df,coords='galactic'):

	cols list(df.columns)
	cols_low = [x.lower() for x in cols]

	if coords=='galactic':
		if ('l' and 'b') in cols_low:
			l_col = cols_low.index('l')
			b_col = cols_low.index('b')
			mu = np.array(df[[cols[l_col],cols[b_col]]])
		elif ('ra' and ('de' or 'dec')) or ('raj2000' and ('dej2000' or 'decj2000')) in cols_low:
			ra_col













