"""
This script simulates the centre of a Hub-Filament System with saturated pixels
"""

### imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd

def gaussian(points,mx,my,N,theta,FWHMx,FWHMy):
	"""
	Gaussian function in 2D:
		- points = (x,y) is the grid array at which the function is being evaluated
		- (mx,my) = (mu_x,mu_y) is the centre of the distribution
		- N is an arbitrary normalization constant
		- FWHM is given in the same units as the 'points' argument
	"""
	sigmax = FWHMx/(2*np.sqrt(2*np.log(2)))
	sigmay = FWHMy/(2*np.sqrt(2*np.log(2)))
	alphax = 1/(2*sigmax**2)
	alphay = 1/(2*sigmay**2)
	x,y = points
	xl = x*np.cos(theta) - y*np.sin(theta)
	yl = x*np.sin(theta) + y*np.cos(theta)
	mxl = mx*np.cos(theta) - my*np.sin(theta)
	myl = mx*np.sin(theta) + my*np.cos(theta)
	z = N * np.exp( - alphax*(xl-mxl)**2 - alphay*(yl-myl)**2 )
	return z

def gaussianMult(points,*args):
	""" Sum multiple 2D gaussian functions. """
	z = 0
	for i in range(len(args)//6):
		mx,my,N,theta,FWHMx,FWHMy = args[6*i:6*(i+1)]
		z += gaussian(points,mx,my,N,theta,FWHMx,FWHMy)
	return z


def fitter(grid,data,input_filename="",
	peaks=1,mu=[],theta=[],FWHM=[],
	units_theta='deg',units_FWHM='arcsec',
	var_pos=0.01,var_theta=0.5,var_FWHM=0.5,
	dist_factor=2,exclude_below=0.1):
	"""
	Function takes array image, its grid and boolean array of same shape,
	which is True where pixels are saturated and False elsewhere.
	Returns the image with saturated pixels corrected.
	Saturated pixels in data can only be represented by 'nan' values.
	"""

	X,Y = grid # unpack grid
	sat = np.isnan(data) # detect saturated pixels
	mu = mu.copy()
	theta = theta.copy()
	FWHM = FWHM.copy()
	xl,xr,yb,yt = X[0,0], X[0,-1], Y[0,0], Y[-1,0] # limits of the image


    # check for an input file with the source parameters, which would have priority
	if input_filename == "":
		# initial guess for peak positions
		if len(mu)==0:
			mu_x = X[sat].mean()
			mu_y = Y[sat].mean()
			mu = np.array(peaks*[[mu_x,mu_y]],float)

        # initial guess for semimajor-axis angle with x-axis
		if len(theta)==0:
			theta = np.zeros(peaks,float)
			var_theta = np.pi
		elif units_theta == 'deg':
			theta *= np.pi/180
			var_theta *= np.pi/180
		elif units_theta == "rad":
			"do nothing"
		else:
			return """Error: theta unit invalid, please enter "rad" for radian or leave blank for degree."""

        # initial guess for Full-Width at Half Maximum
		if len(FWHM)==0:
			FWHM = np.ones([peaks,2],float)*18.2/3600
		elif units_FWHM == 'arcsec':
			FWHM /= 3600
			var_FWHM /= 3600
		elif units_FWHM == "deg":
			"do nothing"
		else:
			return """Error: FWHM unit invalid, please enter "deg" for degree or leave blank for arcsec."""

	else:
		table = fits.open(input_filename)[1].data
		df = pd.DataFrame(table)
		df['l'] = np.empty(8)
		df['b'] = np.empty(8)
		names = df['Name'].copy()
        
        # convert the equatorial coordenates to galactic coordenates adding 2 columns for these
		for i in range(len(df)):
			df.iloc[i,-2] = float(names[i][1:9])
			df.iloc[i,-1] = float(names[i][9:])
        
        # mu in galactic coords now
		mu = np.array(df.iloc[:,-2:],float)
		theta = np.array(df['PA'],float)*np.pi/180
		FWHM = np.array(df.iloc[:,4:6],float)/3600
		var_FWHM /= 3600
        
	# initial guess for peak heights
	N = data[~sat].max()

	# initial guess parameters
	guess_params = np.empty(6*peaks,float)
	guess_params[::6] = mu[:,0]
	guess_params[1::6] = mu[:,1]
	guess_params[2::6] = N*1.1
	guess_params[3::6] = theta
	guess_params[4::6] = FWHM[:,0]
	guess_params[5::6] = FWHM[:,1]

	# upper and lower bounds for parameters to vary
	lower_bounds = guess_params.copy()
	upper_bounds = guess_params.copy()
	var_list = np.array([var_pos,var_pos,0,var_theta,var_FWHM,var_FWHM])

	for i in range(6):
		lower_bounds[i::6] -= var_list[i]
		upper_bounds[i::6] += var_list[i]
	lower_bounds[2::6] = 0
	upper_bounds[2::6] = np.inf

	# limit fitting pixels to the vicinity of the sources
	near_pixels = np.empty(list(X.shape)+[peaks],bool)
	for i in range(peaks):
		near_pixels[:,:,i] = np.sqrt((X-mu[i,0])**2 + (Y-mu[i,1])**2) <= dist_factor*FWHM[i,0]
	near_pixels = near_pixels.any(axis=2)

	# exclude background from fitting pixels
	unsat_vals = np.sort(data[~sat].ravel())
	index = int(exclude_below*len(unsat_vals))
	above_min = data >= unsat_vals[index]

	# processed data points to be fitted
	conditions = (~sat) & near_pixels & above_min
	fit_x = np.array([X[conditions],Y[conditions]])
	fit_data = data[conditions]

	# fitting
	print(guess_params)
	params,cov = curve_fit(gaussianMult,fit_x,fit_data,guess_params,bounds=(lower_bounds,upper_bounds),maxfev=4000)
	
	# generating final, corrected image
	image = gaussianMult((X,Y),*params)
	image[~sat] = data[~sat]
	used_image = image.copy()
	used_image[~conditions] = np.nan

    # generating residuals
	gaussians = gaussianMult(grid,*params)
	residuals = data - gaussians
    
    # plotting results
	plt.figure(figsize=(8,8))
	plt.title("Log10 Image used for fit", fontsize=14)
	plt.imshow(np.log10(used_image),origin='lower',extent=[xl,xr,yb,yt])
	plt.colorbar()   
	plt.show()
    
	plt.figure(figsize=(8,8))
	plt.title("Log10 Final image and source points obtained", fontsize=14)
	plt.imshow(np.log10(image),origin='lower',extent=[xl,xr,yb,yt])
	plt.scatter(params[0:peaks*6:6],params[1:peaks*6:6],color='red',marker='+')
	plt.colorbar()   
	plt.show()
    
	plt.figure(figsize=(8,8))
	plt.title("Residuals image", fontsize=14)
	plt.imshow(residuals,origin='lower',extent=[xl,xr,yb,yt])
	plt.scatter(params[0:peaks*6:6],params[1:peaks*6:6],color='red',marker='+')
	plt.colorbar()   
	plt.show()
    
	return params,image,residuals

def display_fits(file,lims=[],return_vals=False,graph=True):
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

	if graph == True:    
		plt.figure(figsize=(8,8))
		plt.imshow(np.log10(data_sub),origin='lower',extent=(xl,xr,yb,yt))
		plt.xlabel(header['ctype1']+' [{}]'.format(header['cunit1']))
		plt.ylabel(header['ctype2']+' [{}]'.format(header['cunit2']))
		plt.colorbar()
		plt.show()

	if return_vals:
		grid = np.meshgrid(xsub,ysub)
		sat_area = np.isnan(data_sub)
		return grid,data_sub,sat_area

    
def file_fitter(file,FWHMval):
	grid,data,sat_area = display_fits(file,return_vals=True)
    
	if sat_area.any():
		xl = float(input('left '))
		xr = float(input('right '))
		yb = float(input('bottom '))
		yt = float(input('top '))
        
		grid_sub,data_sub,sat_area_sub = display_fits(file,lims=[xl,xr,yb,yt],return_vals=True)

		#FWHM flexibility for extreme ellipses?
		n = int(input("sources "))
		FWHM_flex = float(input("FWHM flexible (0.5 -> default arcsec)? "))
		FWHM_v = float(FWHMval)
            
		if not sat_area.any():
			return "No Saturated Region"
		try:
			params,corrected = fitter(grid_sub,data_sub,sat_area_sub,FWHM=np.array(n*[2*[FWHM_v]]),peaks=n,bg_fitting=True,var_FWHM=FWHM_flex)
			plt.figure(figsize=(8,8))
			plt.imshow(np.log10(corrected),origin = "lower",extent=(xl,xr,yb,yt))
			for i in range(n):
				plt.plot(params[i*6],params[i*6+1],'ro')
			plt.show()
            
			center = (grid_sub[0][sat_area_sub].mean(),grid_sub[1][sat_area_sub].mean())
			radii = np.ravel(np.sqrt((grid_sub[0]-center[0])**2 + (grid_sub[1]-center[1])**2))
            
			gauss = np.zeros(len(radii))
			for i in range(n):
				peak_rad = np.sqrt((params[6*i]-center[0])**2 + (params[6*i+1]-center[1])**2)
				sigmax = params[6*i+4]/(2*np.sqrt(2*np.log(2)))
				alphax = 1/(2*sigmax**2)
				gauss += params[6*i+2]*np.exp(-alphax*(radii-peak_rad)**2)
            
			sort = np.argsort(radii)
			plt.figure(figsize=(10,6))
			plt.plot(radii[sort],np.ravel(data_sub)[sort],'x')
			plt.plot(radii[sort],gauss[sort],'k-')
			plt.show()
            
			return params,corrected
		except:
			return "No fit was found"
	else:
		return "No Saturated Region"

