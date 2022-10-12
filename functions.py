"""This file contains mathematical and other functions used in the rest of the program."""


import numpy as np

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