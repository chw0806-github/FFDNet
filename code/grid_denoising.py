import numpy as np
import torch

from dataloading import compute_std_map
from utils import psnr


def denoise_image_grid(model, img, modality, window_size, const_sigma, grid_size=250):
    '''
    image is a numpy array with shape >= (250,250)
    modality, window_size, const_sigma are parameters used to determine how to compute the std-map
    
    model will be applied in squares of size (grid_size,grid_size) all over the image.
    
    Returns the denoised img
    '''
    denoised_img = np.zeros(img.shape)
    img_free = np.ones(img.shape,dtype=bool) # indicating which parts of the img were NOT yet processed
    model.cuda()
    model.eval()
    # Cut the image in squares of size 'grid_size', loop over them
    for lower_x in range(0,img.shape[1],grid_size):
        if lower_x + grid_size >= img.shape[1]:
            lower_x = img.shape[1]-grid_size
        upper_x = lower_x+grid_size
        for lower_y in range(0,img.shape[0],grid_size):
            if lower_y + grid_size >= img.shape[0]:
                lower_y = img.shape[0]-grid_size
            # Process one square
            upper_y = lower_y+grid_size
            crop = img[lower_y:upper_y,lower_x:upper_x]
            std_map = compute_std_map(crop, None, modality, False, window_size, const_sigma)
            # Convert crop, std_map to correct Torch Cuda Tensors
            crop = np.expand_dims(np.expand_dims(crop,axis=0),axis=0)
            std_map = np.expand_dims(np.expand_dims(std_map,axis=0),axis=0)
            crop = torch.from_numpy(crop).float().cuda()
            std_map = torch.from_numpy(std_map).float().cuda()
            # Denoise the square, store the result
            denoised_crop = model(crop, std_map).cpu().detach().numpy()
            denoised_img[lower_y:upper_y,lower_x:upper_x] = (denoised_img[lower_y:upper_y,lower_x:upper_x] 
                                                            + img_free[lower_y:upper_y,lower_x:upper_x]*denoised_crop)
            img_free[lower_y:upper_y,lower_x:upper_x] = False
    return denoised_img

# compute psnr's of real image denoisals
def get_grid_psnrs(img, denoised_img, gt_img, grid_size=250):
    psnr_list = []
    psnr_noisy_list = []
    for lower_x in range(0,img.shape[1],grid_size):
        if lower_x + grid_size >= img.shape[1]:
            lower_x = img.shape[1]-grid_size
        upper_x = lower_x+grid_size
        for lower_y in range(0,img.shape[0],grid_size):
            if lower_y + grid_size >= img.shape[0]:
                lower_y = img.shape[0]-grid_size
            upper_y = lower_y+grid_size
            crop = img[lower_y:upper_y,lower_x:upper_x]
            denoised_crop = denoised_img[lower_y:upper_y, lower_x:upper_x]
            gt_crop = gt_img[lower_y:upper_y, lower_x:upper_x]
            psnr_list.append(psnr(denoised_crop, gt_crop))
            psnr_noisy_list.append(psnr(crop, gt_crop))
    return psnr_list, psnr_noisy_list