"""
This script simulates the centre of a Hub-Filament System with saturated pixels
"""

### imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits

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


def simulate(N):
	""" Simulate noisy data to fit """
	x = np.linspace(-10,10,N)
	y = x.copy()
	grid = np.meshgrid(x,y)
	image = gaussian(grid,-2,-1,1,0,18.2,18.2)
	image += gaussian(grid,1,2,1.5,0,18.2,18.2)
	image += gaussian(grid,1,-2,1,0,18.2,18.2)
	return grid,image

def fitter(grid,data,sat,mu=[],theta=[],FWHM=[],peaks=1,
	helper_peaks=False,units_theta='deg',units_FWHM='arcsec',
	var_pos=0.01,var_theta=0.5,var_FWHM=0.5):
	"""
	Function takes array image, its grid and boolean array of same shape,
	which is True where pixels are saturated and False elsewhere.
	Returns the image with saturated pixels corrected.
	Saturated pixels in data can only be represented by 'nan' values.
	"""

	X,Y = grid # unpack grid
    
    # initial guess for peak positions
	if len(mu)==0:
		mu_x = X[sat].mean()
		mu_y = Y[sat].mean()
		mu = np.array(peaks*[[mu_x,mu_y]],float)
		mu_given = False
	else:
		mu_given = True

	# initial guess for peak heights
	N = data[np.isnan(data)==False].max()

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

	guess_params = np.empty(6*peaks,float)
	guess_params[::6] = mu[:,0]
	guess_params[1::6] = mu[:,1]
	guess_params[2::6] = N*1.1
	guess_params[3::6] = theta
	guess_params[4::6] = FWHM[:,0]
	guess_params[5::6] = FWHM[:,1]

	lower_bounds = guess_params.copy()
	upper_bounds = guess_params.copy()

	if not mu_given:
		lower_bounds[::6] = X[sat].min()
		lower_bounds[1::6] = Y[sat].min()
		upper_bounds[::6] = X[sat].max()
		upper_bounds[1::6] = Y[sat].max()

	lower_bounds[::6] -= var_pos
	lower_bounds[1::6] -= var_pos
	lower_bounds[2::6] = N
	lower_bounds[3::6] -= var_theta
	lower_bounds[4::6] -= var_FWHM
	lower_bounds[5::6] -= var_FWHM

	upper_bounds[::6] += var_pos
	upper_bounds[1::6] += var_pos
	upper_bounds[2::6] = np.inf
	upper_bounds[3::6] += var_theta
	upper_bounds[4::6] += var_FWHM
	upper_bounds[5::6] += var_FWHM

	# add helper_peaks to the mix
	if helper_peaks:
		FWHMx,FWHMy = FWHM.mean(axis=0)

		data0 = data[1:-1,1:-1]
		prex = np.zeros(data.shape)
		posx = prex.copy()
		prey = prex.copy()
		posy = prex.copy()
		prex[1:-1,1:-1] = data0 - data[1:-1,:-2]
		posx[1:-1,1:-1] = data0 - data[1:-1,2:]
		prey[1:-1,1:-1] = data0 - data[:-2,1:-1]
		posy[1:-1,1:-1] = data0 - data[2:,1:-1]

		bool_maxima = (prex>0)&(posx>0)&(prey>0)&(posy>0)
		maxima = np.array([X[bool_maxima],Y[bool_maxima],data[bool_maxima]]).transpose()
		m = len(maxima[:,0])

		helper_params = np.empty(m*6)
		helper_params[::6] = maxima[:,0]
		helper_params[1::6] = maxima[:,1]
		helper_params[2::6] = maxima[:,2]
		helper_params[3::6] = 0
		helper_params[4::6] = FWHMx
		helper_params[5::6] = FWHMy

		helper_lb = helper_params.copy()
		helper_ub = helper_params.copy()

		helper_lb[::6] -= var_pos
		helper_lb[1::6] -= var_pos
		helper_lb[2::6] *= 0.9
		helper_lb[3::6] -= np.pi
		helper_lb[4::6] = 0
		helper_lb[5::6] = 0

		helper_ub[::6] += var_pos
		helper_ub[1::6] += var_pos
		helper_ub[2::6] *= 1.1
		helper_ub[3::6] += np.pi
		helper_ub[4::6] *= 2
		helper_ub[5::6] *= 2

		guess_params = np.concatenate((guess_params,helper_params))
		lower_bounds = np.concatenate((lower_bounds,helper_lb))
		upper_bounds = np.concatenate((upper_bounds,helper_ub))

	fit_x = np.array([X[sat==False],Y[sat==False]])
	fit_data = data[sat==False]

	params,cov = curve_fit(gaussianMult,fit_x,fit_data,guess_params,bounds=(lower_bounds,upper_bounds),maxfev=4000)
	image = gaussianMult((X,Y),*params)
	image[sat==False] = data[sat==False]
	return params,image

def display_fits(file,lims=[],return_vals=False):
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

	plt.figure(figsize=(8,8))
	plt.imshow(np.log10(data_sub),origin='lower',extent=(xl,xr,yb,yt))
	plt.show()

	if return_vals:
		grid = np.meshgrid(xsub,ysub)
		sat_area = np.isnan(data_sub)
		return grid,data_sub,sat_area



if __name__ == '__main__':

	grid,data = simulate(50)
	x,y = grid[0][0,:],grid[1][:,0]
	sat = data>0.5*data.max()
	ticks = np.arange(0,100)
	labels = np.linspace(-1,1,100)
	plt.imshow(data)
	plt.show()
	plt.imshow(sat)
	plt.show()

	data2 = data.copy()
	data2[sat] = 0
	plt.imshow(data2)
	plt.show()

	params,fit_data = fit(grid,data,sat,FWHM=np.array(3*[2*[18.2]]),peaks=3)

	plt.imshow(fit_data)
	plt.colorbar()
	plt.show()

	plt.imshow(fit_data-data)
	plt.colorbar()
	plt.show()

	print(params[:6])
	print(params[6:12])
	print(params[12:])
    
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
		FWHM_flex = int(input("FWHM flexible (0.5 -> default arcsec)? "))
        
		if not sat_area.any():
			return "No Saturated Region"
		try:
			params,corrected = fitter(grid_sub,data_sub,sat_area_sub,FWHM=np.array(n*[2*[FWHMval]])/3600,peaks=n,helper_peaks=True,var_FWHM=FWHM_flex)
			plt.figure(figsize=(8,8))
			plt.imshow(np.log10(corrected),extent=(xl,xr,yb,yt))
			for i in range(n):
				plt.plot(params[i*6],params[i*6+1],'ro')
			plt.show()
            
			center = (grid_sub[0][sat_area_sub].mean(),grid_sub[1][sat_area_sub].mean())
			radii = np.ravel(np.sqrt((grid_sub[0]-center[0])**2 + (grid_sub[1]-center[1])**2))
            
			sigmax = params[4]/(2*np.sqrt(2*np.log(2)))
			alphax = 1/(2*sigmax**2)
			gauss = params[2]*np.exp(-alphax*radii**2)
            
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
