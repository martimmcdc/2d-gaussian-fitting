"""
This script contains functions which can be called to help visualization.
"""

### imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LogNorm
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit
from astropy.io import fits
from functions import *


def find_edges(bool_array):
	"""This functions identifies the pixels
	which bound a region of True values."""
	xlen,ylen = bool_array.shape
	edges = np.zeros(bool_array.shape,bool)
	neighbours = np.zeros([ylen-2,xlen-2,8],bool)
	neighbours[:,:,0] = ~bool_array[1:-1,:-2] # left
	neighbours[:,:,1] = ~bool_array[1:-1,2:] # right
	neighbours[:,:,2] = ~bool_array[2:,1:-1] # top
	neighbours[:,:,3] = ~bool_array[:-2,1:-1] # bottom
	neighbours[:,:,4] = ~bool_array[:-2,:-2] # bottom-left diag
	neighbours[:,:,5] = ~bool_array[2:,2:] # top-right diag
	neighbours[:,:,6] = ~bool_array[2:,:-2] # top-left diag
	neighbours[:,:,7] = ~bool_array[:-2,2:] # bottom-right diag
	edges[1:-1,1:-1] = bool_array[1:-1,1:-1]&neighbours.any(axis=2)
	return edges

def plot_edge(edge):
	"""This function orders the edge points,
	so they can be plotted as a line."""
	center = edge.mean(axis=1)
	dif = edge.transpose() - center
	dif1 = dif[dif[:,0]>=0,:]
	dif2 = dif[dif[:,0]<0,:]
	tan1 = dif1[:,1]/dif1[:,0]
	tan2 = dif2[:,1]/dif2[:,0]
	angle1 = np.arctan(tan1)
	angle2 = np.arctan(tan2)
	order1 = np.argsort(angle1)
	order2 = np.argsort(angle2)
	edge_points = np.concatenate((dif1[order1],dif2[order2]))+center
	return edge_points

def draw_contour(ax,grid,data,mu,FWHM,fitting_radius,bg_method):
	"""Draw contour of region used for fit onto existing figure."""
	X,Y = grid
	peaks = len(mu)
	sat = np.isnan(data)
	FWHM = FWHM.copy()/3600

	# limit fitting pixels to the vicinity of the sources
	near_pixels = np.empty(list(X.shape)+[peaks],bool)
	for i in range(peaks):
		near_pixels[:,:,i] = np.sqrt((X-mu[i,0])**2 + (Y-mu[i,1])**2) <= fitting_radius*FWHM[i,0]
	near_pixels = near_pixels.any(axis=2)

	# exclude background from fitting pixels
	bg = background(data[~sat&near_pixels].copy(),method=bg_method)
	above_bg = data >= bg

	contour = find_edges(sat | (near_pixels & above_bg))
	edge_points = plot_edge(np.array([X[contour],Y[contour]],float))

	ax.plot(edge_points[:,0],edge_points[:,1])


def sweep_fit(grid,data,gaussians):
	"""Make animated sweeping of the image and give transverse
	perspective of data and fit in linear and logarithmic scales."""
	X,Y = grid
	x,y = X[0,:],Y[:,0]
	xl,xr,yb,yt = x[0],x[-1],y[0],y[-1]
	sat = np.isnan(data)

	fig = plt.figure(figsize=(12,6))
	gs = fig.add_gridspec(2, 2)
	ax0 = fig.add_subplot(gs[:,0],title='Image sweeping')
	ax1 = fig.add_subplot(gs[0,1],title='Linear scale',yscale='linear')
	ax2 = fig.add_subplot(gs[1,1],title='Logarithmic scale',yscale='log')

	fig.tight_layout(pad=1.2)

	ax1.set_ylim(0,gaussians.max())
	ax1.set_xlim(x[0],x[-1])
	ax2.set_xlim(x[0],x[-1])
	ax2.set_ylim(data[~sat].min(),gaussians.max())

	ax0.imshow(data,origin='lower',extent=[xl,xr,yb,yt],norm=LogNorm())
	line0 = ax0.axhline(y[0],color='red')
	line1, = ax1.plot(x,data[0,:],label='Data')
	line2, = ax1.plot(x,gaussians[0,:],label='Fit')
	line3, = ax2.plot(x,data[0,:],label='Data')
	line4, = ax2.plot(x,gaussians[0,:],label='Fit')
	def f(i):
	    line0.set_ydata(y[i])
	    line1.set_ydata(data[i,:])
	    line2.set_ydata(gaussians[i,:])
	    line3.set_ydata(data[i,:])
	    line4.set_ydata(gaussians[i,:])
	    return [line0,line1,line2,line3,line4]
	ax1.legend()
	ax2.legend()
	ani = FuncAnimation(fig,f,interval=200,blit=True,save_count=50)











