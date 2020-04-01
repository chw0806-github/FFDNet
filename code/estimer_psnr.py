"""
Author: emanuele dalsasso
Estimate PSNR for SAR amplitude images
"""
import numpy as np

def psnr(Shat, S):
    # Shat: a SAR amplitude image
    # S:    a reference SAR image
    P = np.quantile(S, 0.99)
    res = 10 * np.log10((P ** 2) / np.mean(np.abs(Shat - S) ** 2))
    return res