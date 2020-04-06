import torch
import os
import numpy as np
from torch.utils.data import IterableDataset
from scipy.signal import convolve2d
from scipy.special import polygamma

modality_amplitude = 'modality_amplitude'
modality_log_intensity = 'modality_log_intensity'

def compute_std_map(noised_img, 
                    ground_truth_img, 
                    modality,     
                    std_from_ground_truth=False, 
                    window_size=7,
                    const_sigma = False,L=1.0,):
    '''
    Computes the noise level map for 'noised_img'
    '''
    
    if modality == modality_amplitude and not const_sigma:
        # compute the mean image (by convolution)
        masque_loc = (np.ones((window_size, window_size)) / (window_size * window_size))
        if std_from_ground_truth:
            ima_int_mean = convolve2d(ground_truth_img, masque_loc, mode='same')
        else:
            ima_int_mean = convolve2d(noised_img, masque_loc, mode='same')
            #Calculate the std (with the mean of the groundtruth)
        sigma_theoretical = ima_int_mean * 0.53
    elif modality == modality_log_intensity and not const_sigma:
        # Log-intensity
        # has constant std
        sigma_theoretical = polygamma(1,float(L))*(np.ones(noised_img.shape))
    elif const_sigma: 
        sigma_theoretical = np.ones(noised_img.shape)
    
    return sigma_theoretical



class MyIterableDataset(IterableDataset):
    '''
    Dataset used for training and testing.
    Loading all images only once into memory (~300MB)
    Outputs random crops from those images
    '''
    
    def __init__(self, 
                 image_npy_paths,
                 modality,
                 noise2noise,
                 crop_shape=(244,244), 
                 transform=None, 
                 train = True, 
                 lower_x_test=None, 
                 lower_y_test=None, 
                 nb_test_images = 20,
                 std_from_ground_truth = False,
                 window_size=7, 
                 const_sigma = False,
                 normalize=False):
        '''
            Args:
                image_npy_paths: list of paths to .npy files to use
                modality: modality_amplitude or modality_log_intensity
                noise2noise: True/False
        '''
        
        super(MyIterableDataset).__init__()
        
        self.image_npy_paths = image_npy_paths
        self.crop_shape = crop_shape
        self.transform = transform
        self.modality = modality
        self.noise2noise = noise2noise
        self.std_from_ground_truth = std_from_ground_truth
        self.window_size = window_size
        self.const_sigma = const_sigma 
        self.choices = {'marais1':190.92, 
                        'marais1_10':190.92,
                        'marais1_11':190.92,
                        'marais2': 168.49, 
                        'marais2_10':168.49,
                        'marais2_11':168.49,
                        'saclay':470.92, 
                        'lely':235.90, 
                        'lely_10':235.90,
                        'lely_11':235.90,
                        'ramb':167.22,
                        'risoul':306.94, 
                        'limagne':178.43,
                        'limagne_10':178.43,
                        'ligmane_20':178.43}
        self.normalize = normalize
        self.normalization_value = 470.0
        self.images = []
        for path in image_npy_paths:
            if os.path.exists(path):
                img = np.load(path)
                if normalize:
                    img = np.clip(img, 0.0, self.normalization_value)/self.normalization_value
                self.images.append(img)
            else:
                print('Could not find',path)
                assert False
        
        # Iterator will return random crops from images.
        # One per image, iterating through the images.
        self.next_img = 0
        self.image_epoch_ratio = 30 #How many images are cropped out of each image --> only important for epoch calculation
        self.test_crop = 0
        self.train = train
        self.lower_x_test = lower_x_test
        self.lower_y_test = lower_y_test
        self.nb_test_images = nb_test_images

    def __iter__(self):
      #Only iterate over each image once in one epoch 
      #return self
      if self.train: 
        for index in range(len(self.images)* self.image_epoch_ratio):
          yield self.__next__()
        
      else: 
        for index in (range(self.nb_test_images)): 
          yield self.__next__()

    def __next__(self):
        img = self.images[self.next_img]
        picture_name = self.image_npy_paths[self.next_img].split("/")[-1].split(".")[0]
        thresh_level = self.choices[picture_name]
       
        if self.train:
          #Random picks of the crops during training  
          lower_x = np.random.randint(0,img.shape[1]-self.crop_shape[1]+1)
          lower_y = np.random.randint(0,img.shape[0]-self.crop_shape[0])
          
        else: 
          #Deterministic picks of the crops during testing 
          lower_x =int(self.lower_x_test[self.test_crop] * (img.shape[1] - self.crop_shape[1])) + 1
          lower_y =int(self.lower_y_test[self.test_crop] * (img.shape[0]  - self.crop_shape[0])) + 1
          self.test_crop = (self.test_crop + 1)%self.nb_test_images
        self.next_img = (self.next_img + 1)%len(self.images)
        upper_x = lower_x + self.crop_shape[1]
        upper_y = lower_y + self.crop_shape[0]
        crop = img[lower_y:upper_y, lower_x:upper_x]
        noised_crop = crop
        noised_crop_2 = None
        if self.transform:
            noised_crop = self.transform(crop)
            if self.noise2noise:
                noised_crop_2 = self.transform(crop)

        # Convert Amplitude -> log intensity
        if self.modality == modality_log_intensity:
            crop = 2*np.log(crop)
            noised_crop = 2*np.log(noised_crop)
            if self.noise2noise:
                noised_crop_2 = 2*np.log(noised_crop_2)

        # Compute std-map
        sigma_theoretical = compute_std_map(noised_crop, crop, modality=self.modality,\
                                            std_from_ground_truth = self.std_from_ground_truth,\
                                            window_size = self.window_size,const_sigma= self.const_sigma)

        # Add channels dimension
        crop = np.expand_dims(crop, axis=0)
        noised_crop = np.expand_dims(noised_crop, axis=0)
        sigma_theoretical = np.expand_dims(sigma_theoretical, axis=0)
        if self.noise2noise:
            noised_crop_2 = np.expand_dims(noised_crop_2, axis=0)
            
        if self.noise2noise:
            return {'noised_crop':noised_crop,
                    'target_crop':noised_crop_2,
                    'std_map':sigma_theoretical,
                    'ground_truth':crop,
                    'thresh':thresh_level}
        else:
            return {'noised_crop':noised_crop, 
                    'target_crop':crop, 
                    'std_map':sigma_theoretical,
                    'thresh':thresh_level}
      
    def __len__(self):
      if self.train: 
        return len(self.images) * self.image_epoch_ratio
      else: 
        return self.nb_test_images
    
    def order_restart(self): 
        self.next_img = 0
        self.test_crop = 0
        
        


class NoiseInjector():
    '''
    Data transformation for the pytorch learning pipeline
    Injects speckle noise into an image in amplitude format
    '''
    
    def __init__(self, L=1, noise_level=1.0):
        self.noise_level = noise_level
        self.L = L

    def __call__(self, img):
        noised_img = NoiseInjector.injectspeckle_amplitude(img,self.L)
        noise_lvl_map = np.ones(noised_img.shape)*self.noise_level
        return np.array(noised_img)

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
