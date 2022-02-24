#!/usr/bin/env python

"""
Display function 
"""
import numpy as np 
from cv2 import cvtColor, COLOR_BGR2RGB
	
def Norma_Xpercentile(image_data, prct:int = 2, BGR2RGB=True):
	'''
	Function that perform x percent histogram equalization of RGB images display
	'''

	a = np.ndarray(image_data.shape, dtype='float32')  
	a[:,:,0] = (image_data[:,:,0] - np.nanpercentile(image_data[:,:,0],prct))/(np.nanpercentile(image_data[:,:,0],100-prct) - np.nanpercentile(image_data[:,:,0],prct))
	a[:,:,1] = (image_data[:,:,1] - np.nanpercentile(image_data[:,:,1],prct))/(np.nanpercentile(image_data[:,:,1],100-prct) - np.nanpercentile(image_data[:,:,1],prct))
	a[:,:,2] = (image_data[:,:,2] - np.nanpercentile(image_data[:,:,2],prct))/(np.nanpercentile(image_data[:,:,2],100-prct) - np.nanpercentile(image_data[:,:,2],prct))
	if BGR2RGB: 
		a = cvtColor(a, COLOR_BGR2RGB)
	return a