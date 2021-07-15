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
	image = gaussian2D(grid,0,0,0.1,0.5,1,0)+gaussian2D(grid,0.5,0.5,0.1,0.1,0.7,0)
	return grid,image#+noise

def fit(grid,data,sat,peaks=1):
	"""
	Function takes array image, its grid and boolean array of same shape,
	which is True where pixels are saturated and False elsewhere.
	Returns the image with saturated pixels corrected.
	"""
	Ndata = np.count_nonzero(sat==False) # number of usable data points
	Nx,Ny = data.shape # number of points in x and y axes
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
			k += 1

	[mx,my,ax,ay,N,C],cov = curve_fit(gaussian2D,fit_x,fit_data,guess_params)
	image = np.empty(data.shape)
	image[sat] = gaussian2D((X[sat],Y[sat]),mx,my,ax,ay,N,C)
	image[sat==False] = data[sat==False]
	return image

if __name__ == '__main__':

	# grid,data = simulate(100)
	# sat = np.zeros(data.shape,bool)
	# for i in range(40,61):
	# 	for j in range(40,61):
	# 		a = np.random.randint(0,2,size=1)
	# 		if a==0:
	# 			b = np.random.randint(0,2,size=1)
	# 			sat[i,j] = bool(b)
	# 		else:
	# 			sat[i,j] = True

	# plt.imshow(data)
	# plt.show()
	# plt.imshow(sat)
	# plt.show()

	# data2 = data.copy()
	# data2[sat] = 0
	# plt.imshow(data2)
	# plt.show()
	# fit_data = fit(grid,data2,sat)

	# plt.imshow(fit_data-data)
	# plt.colorbar()
	# plt.show()

	x = np.linspace(-1,1,101)
	y = x.copy()
	grid = np.meshgrid(x,y)

	mux1,muy1,ax1,ay1,N1,C1 = -0.3,-0.3,10,10,1,0 
	mux2,muy2,ax2,ay2,N2,C2 = +0.3,-0.3,10,10,1,0 
	mux3,muy3,ax3,ay3,N3,C3 = 0,0.6,10,10,1,0 

	z = gaussian2Dmult(grid,mux1,muy1,ax1,ay1,N1,C1,mux2,muy2,ax2,ay2,N2,C2,mux3,muy3,ax3,ay3,N3,C3)
	plt.imshow(z,origin='lower')
	plt.show()






