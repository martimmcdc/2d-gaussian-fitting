"""
This script simulates the centre of a Hub-Filament System with saturated pixels
"""

### imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from functions import *

def fitter(grid,data,peaks=1,mu=[],theta=[],FWHM=[],
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

	return params,image,bg

    
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
			return "uh oh"
	else:
		return "No Saturated Region"