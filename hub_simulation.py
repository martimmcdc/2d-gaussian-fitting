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

def simulate(N):
	""" Simulate noisy data to fit
	"""
	x = np.linspace(-1,1,N)
	y = np.linspace(-1,1,N)
	grid = np.meshgrid(x,y)
	noise = np.random.normal(0,0.1,size=(N,N))
	image = gaussian2D(grid,0,0,0.5,0.5,1,0)
	return grid,image+noise

def fit(grid,data,sat,peaks=1):
	"""
	Function takes array image and boolean array of same shape,
	which is True where pixels are saturated and False elsewhere.
	Returns the image with saturated pixels corrected.
	"""
	Ndata = np.count_nonzero(sat==False) # number of usable data points
	Nx,Ny = data.shape # number of points in x and y axes
	x = np.arange(0,Nx) # (x indices) = (x position)*scale + translation
	y = np.arange(0,Ny) # (y indices) = (y position)*scale + translation
	X,Y = grid # index grid

	mu_x = np.floor(X[sat].mean())
	mu_y = np.floor(Y[sat].mean())
	sigma_x = X[sat].max() - X[sat].min()
	sigma_y = Y[sat].max() - Y[sat].min()
	N,C = data.max(),data.min()
	guess_params = np.array([mu_x,mu_y,1/sigma_x**2,1/sigma_y**2,N,C])

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

	[mx,my,ax,ay,N,C],cov = curve_fit(gaussian2D,fit_x,fit_data,guess_params)
	print(cov)
	return gaussian2D(grid,mx,my,ax,ay,N,C)

if __name__ == '__main__':

	grid,data = simulate(100)
	sat = np.zeros(data.shape,bool)
	for i in range(40,61):
		for j in range(40,61):
			a = np.random.randint(0,2,size=1)
			if a==0:
				b = np.random.randint(0,2,size=1)
				sat[i,j] = bool(b)
			else:
				sat[i,j] = True

	plt.imshow(data)
	plt.show()
	plt.imshow(sat)
	plt.show()

	data[sat] = 0
	plt.imshow(data)
	plt.show()
	fit_data = fit(grid,data,sat)

	plt.imshow(fit_data)
	plt.show()






