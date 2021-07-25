"""
This script simulates the centre of a Hub-Filament System with saturated pixels
"""

### imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian2D(points,mx,my,ax,ay,N,C):
	"""
	Gaussian function in 2D:
		- points = (x,y) is the grid array at which the function is being evaluated
		- (mx,my) = (mu_x,mu_y) is the centre of the distribution
		- (ax,ay) = (alpha_x,alpha_y) = [1/(2 * sigma_x^2),1/(2 * sigma_y^2)]
		- N is an arbitrary normalization constant
		- C is a baseline constant
	"""
	x,y = points
	z = N * np.exp( - ax*(x-mx)**2 - ay*(y-my)**2 ) + C
	return z

def f(x,*args):
	return(args[::2])

def gaussian2Dmult(points,*args):
	z = 0
	for i in range(len(args)//6):
		mx,my,ax,ay,N,C = args[6*i:6*(i+1)]
		z += gaussian2D(points,mx,my,ax,ay,N,C)
	return z


def simulate(N):
	""" Simulate noisy data to fit
	"""
	x = np.linspace(-1,1,N)
	y = np.linspace(-1,1,N)
	grid = np.meshgrid(x,y)
	noise = np.random.normal(0,0.01,size=(N,N))
	image = gaussian2D(grid,-0.1,-0.2,10,5,1,0)+gaussian2D(grid,0.3,0.2,10,6,1,0)
	return grid,image#+noise

def fit(grid,data,sat,peaks=1,mu=None,FWHM=None):
	"""
	Function takes array image, its grid and boolean array of same shape,
	which is True where pixels are saturated and False elsewhere.
	Returns the image with saturated pixels corrected.
	"""
	Ndata = np.count_nonzero(sat==False) # number of usable data points
	Nx,Ny = data.shape # number of points in x and y axes
	X,Y = grid # index grid

	if mu==None:
		mu_x = np.floor(X[sat].mean())
		mu_y = np.floor(Y[sat].mean())
	else:
		mu_x,mu_y = mu

	if FWHM==None:
		sigma_x = X[sat].max() - X[sat].min()
		sigma_y = Y[sat].max() - Y[sat].min()
	else:
		sigma_x,sigma_y = FWHM/np.sqrt(8*np.log(2))
	N,C = data.max(),data.min()
	peak_params = np.array([mu_x,mu_y,1/sigma_x**2,1/sigma_y**2,N,C])
	guess_params = peak_params.copy()
	for i in range(1,peaks):
		var = np.random.normal(size=6)
		guess_params = np.concatenate((guess_params,peak_params))

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

	params,cov = curve_fit(gaussian2Dmult,fit_x,fit_data,guess_params)
	image = np.empty(data.shape)
	image[sat] = gaussian2Dmult((X[sat],Y[sat]),*params)
	image[sat==False] = data[sat==False]
	return params,image

if __name__ == '__main__':

	grid,data = simulate(100)
	x,y = grid[0][0,:],grid[1][:,0]
	sat = data>0.95*data.max()
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
	params,fit_data = fit(grid,data,sat,peaks=2)

	plt.imshow(fit_data)
	plt.plot(50+50*params[0],50+50*params[1],'o')
	plt.plot(50+50*params[6],50+50*params[7],'o')
	plt.colorbar()
	plt.show()

	plt.imshow(fit_data-data)
	plt.colorbar()
	plt.show()





