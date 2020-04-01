#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:56:07 2018

@author: emasasso
"""

from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import gamma

working_dir = "./prova/"
test_files = glob(working_dir+'*.npy')

choices = {'marais1':190.92, 'marais2': 168.49, 'saclay':470.92, 'lely':235.90, 'ramb':167.22, 'risoul':306.94, 'limagne':178.43}
for filename in test_files:
    dim = np.load(filename)
    dim = np.squeeze(dim)
    for x in choices:
        if x in filename:
            threshold = choices.get(x)

    dim = np.clip(dim,0,threshold)
    dim = dim/threshold*255
    dim = Image.fromarray(dim.astype('float64')).convert('L')
    imagename = filename.replace("npy","png")
    dim.save(imagename)



def plot_speckle_histogram(pic, percentage=0.9999):
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    flat = pic.flatten()
    filtered = sorted(flat)[:int(percentage*len(flat))]
    count, bins, ignored = ax.hist(filtered,bins=50,label='histogram',density=True)
    y = gamma.pdf(bins,1.0)
    ax.plot(bins,y,label='theoretical')
    ax.set_title('Ratio histogram',fontsize=16)
    ax.set_xlabel('Ratio',fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Frequency',fontsize=14)
    ax.legend(fontsize=12)
    ax.set_yticks([],[])
    fig.tight_layout()
    return fig