import torch
import numpy as np
from scipy.special import polygamma, digamma

####################
## Loss functions ##
####################
def amplitude_l1_loss(out_R, tgt_R):
    return torch.nn.functional.l1_loss(out_R, tgt_R)

def amplitude_l2_loss(out_R, tgt_R):
    '''
    tgt_R should be the ground truth amplitude (unbiased)
    out_R should be the prediction for the unbiased amplitude
    '''
    return torch.nn.functional.mse_loss(out_R, tgt_R)

def log_intensity_l1_loss(out_logR, tgt_logR):
    return torch.nn.functional.l1_loss(out_logR, tgt_logR)
    
def log_intensity_l2_loss(out_logR, tgt_logR):
    '''
    tgt_logR should be the ground truth log intensity (unbiased)
    out_logR should be the prediction for the unbiased log intensity
    '''
    return torch.nn.functional.mse_loss(out_logR, tgt_logR)

def noise2noise_log_intensity_speckle_loss(out_logR, tgt_logR):
    '''
    p(log(I)|log(R)) is Fisher-Tippet distributed
    We use as loss function:
    -ln(p(log(I) = tgt_logR | log(R) = out_logR))
    '''
    #L=1.0
    all_pixel_losses = -(tgt_logR-out_logR - torch.exp(tgt_logR-out_logR))
    return torch.mean(all_pixel_losses)

def noise2noise_amplitude_l2_loss(out_R, tgt_A):
    '''
    debiased_tgt_A = tgt_A*sqrt(4/pi) to predict the unbiased amplitude
    '''
    debiased_tgt = tgt_A*np.sqrt(4/np.pi)
    return torch.nn.functional.mse_loss(out_R, debiased_tgt)
    pass

def noise2noise_log_intensity_l2_loss(out_logR, tgt_logR, L=1.0):
    '''
    p(log(I) | R) is Fisher-Tippet
    tgt_logR should be the logarithm of a noised intensity image.
    '''
    bias = -np.log(L) + digamma(L)
    debiased_tgt = tgt_logR - bias
    return torch.nn.functional.mse_loss(out_logR, debiased_tgt)
