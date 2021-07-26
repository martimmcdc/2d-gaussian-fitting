"""
This script simulates the centre of a Hub-Filament System with saturated pixels
"""

### imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian2D(points,mx,my,N):
	"""
	Gaussian function in 2D:
		- points = (x,y) is the grid array at which the function is being evaluated
		- (mx,my) = (mu_x,mu_y) is the centre of the distribution
		- (ax,ay) = (alpha_x,alpha_y) = [1/(2 * sigma_x^2),1/(2 * sigma_y^2)]
		- N is an arbitrary normalization constant
		- C is a baseline constant
	"""
	FWHM = 3*18.2/18 # pixels
	sigma = FWHM/(2*np.sqrt(2*np.log(2)))
	alpha = 1/(2*sigma**2)
	x,y = points
	z = N * np.exp( - alpha*((x-mx)**2 + (y-my)**2) )
	return z

def gaussian2Dmult(points,*args):
	""" Sum multiple 2D gaussian functions. """
	z = 0
	for i in range(len(args)//3):
		mx,my,N = args[3*i:3*(i+1)]
		z += gaussian2D(points,mx,my,N)
	return z


def simulate(N):
	""" Simulate noisy data to fit """
	x = np.linspace(-10,10,N)
	y = x.copy()
	grid = np.meshgrid(x,y)
	noise = np.random.normal(0,1,size=(N,N))
	image = gaussian2D(grid,-2,-1,1)+gaussian2D(grid,1,2,1.5)+gaussian2D(grid,1,-2,1)
	return grid,image#+noise

def fit(grid,data,sat,peaks=1):
	"""
	Function takes array image, its grid and boolean array of same shape,
	which is True where pixels are saturated and False elsewhere.
	Returns the image with saturated pixels corrected.
	Saturated pixels in data can be represented by both 'nan' and 0 (zero) values.
	"""
	Ndata = np.count_nonzero(sat==False) # number of usable data points
	Nx,Ny = data.shape # number of points in x and y axes
	X,Y = grid # index grid

	mu_x = np.floor(X[sat].mean())
	mu_y = np.floor(Y[sat].mean())

	FWHM = 3*18.2/18

	N = data[np.isnan(data)==False].max()

	peak_params = np.array([mu_x,mu_y,N])
	guess_params = peak_params.copy()
	for i in range(1,peaks):
		var = FWHM*np.random.normal(size=3)
		guess_params = np.concatenate((guess_params,peak_params+var))


	fit_x = np.empty([2,Ndata],float)
	fit_data = np.empty(Ndata,float)
	k = 0
	for i in range(Nx):
		for j in range(Ny):
			if sat[i,j]:
				continue
			else:
				fit_x[:,k] = np.array([X[i,j],Y[i,j]])
				fit_data[k] = data[i,j]
			k += 1

	params,cov = curve_fit(gaussian2Dmult,fit_x,fit_data,guess_params,maxfev=4000)
	image = gaussian2Dmult((X,Y),*params)
	image[sat==False] = data[sat==False]
	return params,image

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

	params,fit_data = fit(grid,data,sat,peaks=3)

	plt.imshow(fit_data)
	plt.colorbar()
	plt.show()

	plt.imshow(fit_data-data)
	plt.colorbar()
	plt.show()

	print(params[:3])
	print(params[3:6])
	print(params[6:])
