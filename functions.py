"""This file contains mathematical and other functions used in the rest of the program."""


### Imports ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord



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



### Fitting ###

def fitter(data,grid,peaks=1,mu=[],theta=[],FWHM=[],
	units_theta='deg',units_FWHM='arcsec',
	var_pos=0.01,var_theta=0.5,var_FWHM=0.5,
	fitting_radius=4,bg_method='hist'):
	"""
	Function takes array image, its grid.
	Returns the image with saturated pixels corrected.
	Saturated pixels in data can only be represented by 'nan' values.
	"""

	X,Y = grid # unpack grid
	sat = np.isnan(data) # detect saturated pixels
	
	# use copies of peak info, because they may be changed in unit conversion
	theta = theta.copy()
	FWHM = FWHM.copy()
    
    # initial guess for peak positions
	if len(mu)==0:
		mu_x = X[sat].mean()
		mu_y = Y[sat].mean()
		mu = np.array(peaks*[[mu_x,mu_y]],float)
	else:
		peaks = len(mu)

	# initial guess for peak heights
	N = data[~sat].max()

	# initial guess for semimajor-axis angle with x-axis
	if len(theta)==0:
		theta = np.zeros(peaks,float)
		var_theta = np.pi
	elif units_theta == 'deg':
		theta *= np.pi/180
		var_theta *= np.pi/180

	# initial guess for Full-Width at Half Maximum
	if len(FWHM)==0:
		FWHM = np.ones([peaks,2],float)*18.2/3600
	elif units_FWHM == 'arcsec':
		FWHM /= 3600
		var_FWHM /= 3600

	# limit fitting pixels to the vicinity of the sources
	near_pixels = np.empty(list(X.shape)+[peaks],bool)
	for i in range(peaks):
		near_pixels[:,:,i] = np.sqrt((X-mu[i,0])**2 + (Y-mu[i,1])**2) <= fitting_radius*FWHM[i,0]
	near_pixels = near_pixels.any(axis=2)

	# exclude background from fitting pixels
	bg = background(data[~sat&near_pixels].copy(),method=bg_method)
	above_bg = data >= bg

	# processed data points to be fitted
	conditions = (~sat) & near_pixels & above_bg
	fit_x = np.array([X[conditions],Y[conditions]])
	fit_data = data[conditions] - bg

	# initial guess parameters
	guess_params = np.empty(6*peaks,float)
	guess_params[::6] = mu[:,0]
	guess_params[1::6] = mu[:,1]
	guess_params[2::6] = N*1.1 - bg
	guess_params[3::6] = theta
	guess_params[4::6] = FWHM[:,0]
	guess_params[5::6] = FWHM[:,1]

	# upper and lower bounds for parameters
	lower_bounds = guess_params.copy()
	upper_bounds = guess_params.copy()
	var_list = np.array([var_pos,var_pos,0,var_theta,var_FWHM,var_FWHM])
	for i in range(6):
		lower_bounds[i::6] -= var_list[i]
		upper_bounds[i::6] += var_list[i]
	lower_bounds[2::6] = 0
	upper_bounds[2::6] = np.inf

	# fitting
	params,cov = curve_fit(gaussianMult,fit_x,fit_data,guess_params,
		bounds=(lower_bounds,upper_bounds),maxfev=4000)
	
	# generating final, corrected image
	image = gaussianMult((X,Y),*params) + bg
	image[~sat] = data[~sat]

	return image,params,bg



#### File and data handeling functions ###

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
	xlim,ylim = (x<=xl)&(x>=xr),(y>=yb)&(y<=yt)
	xsub,ysub = x[xlim],y[ylim]
	data_sub = data[ylim,:][:,xlim]

	# If image is to be displayed
	if show:
		plt.figure(figsize=(8,8))
		plt.imshow(np.log10(data_sub),origin='lower',extent=(xl,xr,yb,yt))
		plt.xlabel(header['ctype1']+' [{}]'.format(header['cunit1']))
		plt.ylabel(header['ctype2']+' [{}]'.format(header['cunit2']))
		plt.show()

	grid = np.meshgrid(xsub,ysub)
	return data_sub,grid

def open_fits_table(file,ext=1):
	"""Open table in .fits format, which is usually in the 1st extension."""

	# Open and read
	hdulist = fits.open(file)
	hdu = hdulist[ext]
	header = hdu.header
	table = hdu.data

	# Return pandas DataFrame() instance
	return pd.DataFrame(table)

def get_parameters(df,coords='galactic',wavelength=''):

	# Identify columns in DataFrame
	cols = list(df.columns)
	cols_low = [x.lower() for x in cols] # list of lowercase column names

	# Lists of possible nomenclature for coordinates (names1),
	# semi-major axis orientation relative to horizontal (names2)
	# and FWHM values (names3)
	names1 = np.array([
		('l','b','galactic'),
		('ra','dec','icrs'),
		('ra','de','icrs'),
		('raj2000','dej2000','icrs'),
		('raj2000','decj2000','icrs')])
	names2 = np.array(['pa','theta','angle'])
	names3 = np.array([
		('amaj'+wavelength,'amin'+wavelength),
		('fwhmx'+wavelength,'fwhmy'+wavelength)])

	# Figure out used nomeclature and return arrays with values for each source
	i = None
	for n in range(len(names1)):
		if (names1[n,0] and names1[n,1]) in cols_low:
			i = n
			break
		else: continue
	if i is None:
		print('Coordinate column names do not correspond to ICRS or Galactic systems.')
		mu = []
	else:
		xcol,ycol = cols_low.index(names1[i,0]),cols_low.index(names1[i,1])
		x,y = df[[cols[xcol],cols[ycol]]].values.transpose()
		xvar,yvar = names1[names1[:,-1]==coords][0,:-1]
		mu = SkyCoord(x*u.degree,y*u.degree,frame=names1[i,2])
		lon = getattr(getattr(mu,coords),xvar).degree
		lat = getattr(getattr(mu,coords),yvar).degree
		mu = np.array([lon,lat],float).transpose()

	i = None
	for n in range(len(names2)):
		if names2[n] in cols_low:
			i = n
			break
		else: continue
	if i is None:
		print('PA column names are not identifiable.')
		theta = []
	else:
		col = cols_low.index(names2[i])
		theta = df[cols[col]].to_numpy(float)

	i = None
	for n in range(len(names3)):
		if (names3[n,0] and names3[n,1]) in cols_low:
			i = n
			break
		else: continue
	if i is None:
		print('FWHM column names are not identifiable.')
		FWHM = []
	else:
		xcol,ycol = cols_low.index(names3[i,0]),cols_low.index(names3[i,1])
		FWHM = df[[cols[xcol],cols[ycol]]].to_numpy(float)
	return mu,theta,FWHM

def grid_lims(grid):
	xl,xr,yb,yt = grid[0][0,0], grid[0][0,-1], grid[1][0,0], grid[1][-1,0]
	return xl,xr,yb,yt

















