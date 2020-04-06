import torch
import os
import mvalab
import basicblock as B
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
from plotimages import plot_speckle_histogram
from utils import weights_init_kaiming, svd_orthogonalization, psnr
from dataloading import modality_amplitude, modality_log_intensity


class FFDNet(nn.Module):
    '''
    Implementation of the FFDNet from the paper
    FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising
            Kai Zhang, Wangmeng Zuo, Lei Zhang
    Implementation of the Network architectur adapted from https://github.com/cszn/KAIR
    '''
    
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2

        self.m_down = B.PixelUnShuffle(upscale_factor=sf)

        m_head = B.conv(in_nc*sf*sf+1, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc*sf*sf, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x = self.m_down(x)
        m = self.m_down(sigma)
        m = torch.mean(m, dim = 1,  keepdim = True) #Take the mean 
        #print("Size of m {}".format(m.size()))
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        #m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)
        
        x = x[..., :h, :w]
        return x