"""
This script contains functions which can be called to help visualization.
"""

### imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from hub_tools import *


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

def draw_contour(grid,data,ax=None):

	X,Y = grid

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

	if ax == None:
		plt.plot(edge_points)
	else:
		ax.plot(edge_points)











