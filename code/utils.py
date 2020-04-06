"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc

File taken from https://github.com/cszn/KAIR

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import subprocess
import math
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.measure.simple_metrics import compare_psnr

def weights_init_kaiming(lyr):
	r"""Initializes weights of the model according to the "He" initialization
	method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
	This function is to be called by the torch.nn.Module.apply() method,
	which applies weights_init_kaiming() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
			clamp_(-0.025, 0.025)
		nn.init.constant(lyr.bias.data, 0.0)





def svd_orthogonalization(lyr):
	r"""Applies regularization to the training by performing the
	orthogonalization technique described in the paper "FFDNet:	Toward a fast
	and flexible solution for CNN based image denoising." Zhang et al. (2017).
	For each Conv layer in the model, the method replaces the matrix whose columns
	are the filters of the layer by new filters which are orthogonal to each other.
	This is achieved by setting the singular values of a SVD decomposition to 1.

	This function is to be called by the torch.nn.Module.apply() method,
	which applies svd_orthogonalization() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		weights = lyr.weight.data.clone()
		c_out, c_in, f1, f2 = weights.size()
		dtype = lyr.weight.data.type()

		# Reshape filters to columns
		# From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
		weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

		# Convert filter matrix to numpy array
		weights = weights.cpu().numpy()

		# SVD decomposition and orthogonalization
		mat_u, _, mat_vh = np.linalg.svd(weights, full_matrices=False)
		weights = np.dot(mat_u, mat_vh)

		# As full_matrices=False we don't need to set s[:] = 1 and do mat_u*s
		lyr.weight.data = torch.Tensor(weights).view(f1, f2, c_in, c_out).\
			permute(3, 2, 0, 1).type(dtype)
	else:
		pass



#############
#The following functions are used from the FFDNet code of Mr. emanuele dalsasso
#############

def psnr(Shat, S):
	"""
	Author: emanuele dalsasso
	Estimate PSNR for SAR amplitude images
	"""
    # Shat: a SAR amplitude image
    # S:    a reference SAR image
	P = np.quantile(S, 0.99)
	res = 10 * np.log10((P ** 2) / np.mean(np.abs(Shat - S) ** 2))
	return res


def injectspeckle_amplitude(img,L):
	"""
	Author: emanuele dalsasso
	Inject speckle in amplitude image
	"""
	rows = img.shape[0]
	columns = img.shape[1]
	s = np.zeros((rows, columns))
	for k in range(0,L):
		gamma = np.abs( np.random.randn(rows,columns) + np.random.randn(rows,columns)*1j )**2/2
		s = s + gamma
	s_amplitude = np.sqrt(s/L)
	ima_speckle_amplitude = np.multiply(img,s_amplitude)
	return ima_speckle_amplitude