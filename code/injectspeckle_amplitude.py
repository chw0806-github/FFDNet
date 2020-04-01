#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:12:23 2018

@author: emasasso
"""
import numpy as np

def injectspeckle_amplitude(img,L):
    rows = img.shape[0]
    columns = img.shape[1]
    s = np.zeros((rows, columns))
    for k in range(0,L):
        gamma = np.abs( np.random.randn(rows,columns) + np.random.randn(rows,columns)*1j )**2/2
        s = s + gamma
    s_amplitude = np.sqrt(s/L)
    ima_speckle_amplitude = np.multiply(img,s_amplitude)
    return ima_speckle_amplitude

im = np.load('denoised_lely.npy')
speckled_image = injectspeckle_amplitude(im,1)
np.save('noisy_lely.npy',speckled_image)
