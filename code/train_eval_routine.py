import os
import mvalab
import FFDNet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import basicblock as B


from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
from plotimages import plot_speckle_histogram
from utils import weights_init_kaiming, svd_orthogonalization, psnr
from dataloading import modality_amplitude, modality_log_intensity


def stats_without_outliers(array):
    '''
    Calculate the mean of the smallest 99% entries of the array.
    Highest 1% are excluded to exclude outliers
    '''
    without_outliers = sorted(array)[:int(len(array)*0.9999)]
    return np.mean(without_outliers), np.var(without_outliers)
    


def train_model(loader_train, loader_test, criterion, epochs, name=''):
    '''
    Function to train an FFDNet model from scratch
    args:
        loader_train: Pytorch Dataloader to load speckled images
        loader_test: Pytorch Dataloader used for validation
        criterion: Loss function used in training
        epochs: Number of epochs that will be trained
        modality: Are the input images in amplitude or log_intensity? MyIterableDataset.modality
        name: string, used as name of logging and output folder
        
    Returns:
        Epoch history of mean PSNR and loss for train and validation
        Checkpoints of the model are logged on hard disk (folder=name)
    '''
    assert loader_train.dataset.noise2noise == loader_test.dataset.noise2noise
    noise2noise = loader_train.dataset.noise2noise
    modality = loader_train.dataset.modality

    #!!!!!!!!!!!!!!!!!!
    save_images = False
    #!!!!!!!!!!!!!!!!!!
    gray = True
    log_dir = "../data/checkpoints/ckpts_{}".format(name)

    resume_training = False
    lr = 1e-5
    no_orthog = False
    save_every = 10
    save_every_epochs = 5
    psnr_list_train = []
    psnr_list_test = []
    loss_list_train = []
    loss_list_test = []

    # Load dataset
    print('> Loading dataset ...')

    #loader_train = dl_train 
    #loader_test = dl_test
    print("\t# of training samples: %d\n" % int(len(loader_train)))

    # Init loggers
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create model
    if not gray:
        in_ch = 3
    else:
        in_ch = 1
    net = FFDNet.FFDNet(in_nc=in_ch)
    # Initialize model with He init
    net.apply(weights_init_kaiming)
    # Define loss
    #criterion = nn.MSELoss(size_average=True)

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    training_params = {}
    training_params['step'] = 0
    training_params['current_lr'] = lr
    training_params['no_orthog'] = no_orthog

    # Training
    for epoch in range(start_epoch, epochs):
        
        # train
        loss_train_epoch = []
        psnr_train_epoch = []
        for i, data in enumerate(loader_train, 0):
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # inputs: noise and noisy image
            img_noised = torch.from_numpy(np.expand_dims(data['noised_crop'][:,0],axis=1)).type(torch.DoubleTensor)
            img_target = data['target_crop']
            if noise2noise:
                img_gt = data['ground_truth']
            sigma_theoretical = data['std_map']

            img_target = Variable(img_target.type(torch.FloatTensor).cuda())
            img_noised = Variable(img_noised.type(torch.FloatTensor).cuda())
            sigma_theoretical = Variable(sigma_theoretical.type(torch.FloatTensor).cuda())
            
            # Evaluate model and optimize it
            out = model(img_noised, sigma_theoretical)
            loss = criterion(out, img_target)
            loss.backward()
            optimizer.step()
            
            # log train loss, psnr
            loss_train_epoch.append(loss.detach().cpu().numpy())
            img_noised = img_noised.detach().cpu().numpy()
            img_out = out.detach().cpu().numpy()
            img_target = img_target.detach().cpu().numpy()
            if noise2noise:
                img_gt = img_gt.detach().cpu().numpy()
            # Convert back to amplitude
            if modality == modality_log_intensity:
                img_noised = np.exp(0.5*img_noised)
                img_out = np.exp(0.5*img_out)
                img_target = np.exp(0.5*img_target)
                if noise2noise:
                    img_gt = np.exp(0.5*img_gt)
            # Invert normalization if necessary
            if loader_train.dataset.normalize:
                img_noised *= loader_train.dataset.normalization_value
                img_out *= loader_train.dataset.normalization_value
                img_target *= loader_train.dataset.normalization_value
                if noise2noise:
                    img_gt *= loader_train.dataset.normalization_value
            if noise2noise:
                psnr_train_epoch.append(psnr(img_out, img_gt))
            else:
                psnr_train_epoch.append(psnr(img_out, img_target))

            training_params['step'] += 1

        # The end of each epoch
        model.eval()
        print("Epoch {}".format(epoch))
        # Compute mean train loss, psnr
        loss_train_epoch = np.mean(loss_train_epoch)
        loss_list_train.append(loss_train_epoch)
        psnr_train_epoch = np.mean(psnr_train_epoch)
        psnr_list_train.append(psnr_train_epoch)
        print("Train Loss : {}".format(loss_train_epoch))
        print("Train PSNR Denoised: {}".format(psnr_train_epoch))
        # Compute Test loss, psnr
        psnr_test_epoch, _, loss_test_epoch, ratio_test_epoch = eval(model,loader_test,criterion, save_img_flag=False)
        loss_list_test.append(loss_test_epoch)
        psnr_list_test.append(psnr_test_epoch)
        
        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        torch.save(model.state_dict(), os.path.join(log_dir, 'net.pth'))
        save_dict = { \
            'state_dict': model.state_dict(), \
            'optimizer' : optimizer.state_dict(), \
            'training_params': training_params, \
            #'args': args\
            }
        
        torch.save(save_dict, os.path.join(log_dir, 'ckpt.pth'))
        if epoch % save_every_epochs == 0:
            torch.save(save_dict, os.path.join(log_dir, \
                                    'ckpt_e{}.pth'.format(epoch+1)))
				    # Apply regularization by orthogonalizing filters
            if not training_params['no_orthog']:
              model.apply(svd_orthogonalization)
        
        del save_dict
    return psnr_list_train, loss_list_train, psnr_list_test, loss_list_test



def eval(model, dl_test, criterion, save_img_flag = False, plot_img_flag = False, save_path_root = None, directory = None):
    '''
    Function to evaluate an FFDNet model
    args:
        model: the FFDNet model
        dl_test: a dataloader of the evaluation data
        criterion: the loss function (for logging)
        
    Returns:
        mean psnr (noisy)
        mean psnr (denoised)
        mean loss
        mean ratio (input/denoised)
    '''
    
    model.eval() 
    if  not save_path_root is None and not directory is None: 
      complete_path = os.path.join(save_path_root, directory)
      if not os.path.exists(os.path.join(save_path_root, directory)):
          os.makedirs(complete_path)
      
    
    batch_size = 1
    noise2noise = dl_test.dataset.noise2noise
    modality = dl_test.dataset.modality
    psnr_list = []
    psnr_noisy_list = []
    mean_ratio_list = []
    std_ratio_list = []
    loss_test_list = []
    for i, data in enumerate(dl_test, 0):
        # Pre-training step
        
        img_noised = torch.from_numpy(np.expand_dims(data['noised_crop'][:,0],axis=1)).type(torch.DoubleTensor)
        img_target = data['target_crop']
        if noise2noise:
            img_gt = data['ground_truth']
        sigma_theoretical = data['std_map']
        threshold_level = data['thresh']
        
        img_target = Variable(img_target.type(torch.FloatTensor).cuda())
        img_noised = Variable(img_noised.type(torch.FloatTensor).cuda())
        sigma_theoretical = Variable(sigma_theoretical.type(torch.FloatTensor).cuda())

        # Evaluate model
        out = model(img_noised, sigma_theoretical)
        loss = criterion(out, img_target)
        loss_test_list.append(loss.detach().cpu().numpy())

        img_in = img_noised.cpu().detach()
        img_out = out.cpu().detach()
        img_target = img_target.cpu().detach()
        if noise2noise:
            img_gt = img_gt.cpu().detach()
        
        # Convert back to amplitude
        if modality == modality_log_intensity:
            img_in = torch.exp(0.5*img_in)
            img_target = torch.exp(0.5*img_target)
            img_out = torch.exp(0.5*img_out)
            if noise2noise:
                img_gt = torch.exp(0.5*img_gt)
        
        # Invert normalization if necessary
        if dl_test.dataset.normalize:
            img_in *= dl_test.dataset.normalization_value
            img_target *= dl_test.dataset.normalization_value
            img_out *= dl_test.dataset.normalization_value
            img_gt *= dl_test.dataset.normalization_value

        # Compute PSNR
        if noise2noise:
            # In noise2noise, img_target is a noised version of the ground truth
            psnr_noisy_list.append(psnr(img_in.numpy(), img_gt.numpy()))
            psnr_list.append(psnr(img_out.numpy(), img_gt.numpy()))
        else:
            psnr_noisy_list.append(psnr(img_in.numpy(), img_target.numpy()))
            psnr_list.append(psnr(img_out.numpy(), img_target.numpy()))
        
        # Intensity ratio
        res_noise_deep = np.square(img_in.detach().cpu().numpy()) / np.square(img_out.detach().cpu().numpy())
        
        if plot_img_flag: 
            mvalab.visusar(img_in.detach().cpu().numpy()[0,0])
            mvalab.visusar(img_out.detach().cpu().numpy()[0,0])
            mvalab.visusar(img_target.detach().cpu().numpy()[0,0])
            mvalab.visusar(res_noise_deep[0,0])
            
        mean_ratio, std_ratio = stats_without_outliers(res_noise_deep.flatten())

        mean_ratio_list.append(mean_ratio)
        std_ratio_list.append(std_ratio)
        mean_deep = np.mean(res_noise_deep)
        var_deep = np.var(res_noise_deep)
        #print("mean : {}, Var: {}".format(mean_deep, var_deep))

        
        if save_img_flag: 
            #Save images
            #Save the input image
            input_img_file = os.path.join(complete_path,"{}_input.png".format(i))
            dim = np.clip(img_in.detach().cpu().numpy()[0,0],0,threshold_level)
            dim = dim/threshold_level*255
            dim = Image.fromarray(dim.numpy().astype('float64')).convert('L')
            dim.save(input_img_file)

            #Output
            output_img_file = os.path.join(complete_path,"{}_output.png".format(i))
            dim = np.clip(img_out.detach().cpu().numpy()[0,0],0,threshold_level)
            dim = dim/threshold_level*255
            dim = Image.fromarray(dim.numpy().astype('float64')).convert('L')
            dim.save(output_img_file)


            #Target
            gt_img_file = os.path.join(complete_path,"{}_Target.png".format(i))
            dim = np.clip(img_target.detach().cpu().numpy()[0,0],0,threshold_level)
            dim = dim/threshold_level*255
            dim = Image.fromarray(dim.numpy().astype('float64')).convert('L')
            dim.save(gt_img_file)

            if noise2noise:
                #Save Ground Truth image 
                gt_img_file = os.path.join(complete_path,"{}_GT_N2N.png".format(i))
                dim = np.clip(img_gt.detach().cpu().numpy()[0,0],0,threshold_level)
                dim = dim/threshold_level*255
                dim = Image.fromarray(dim.numpy().astype('float64')).convert('L')
                dim.save(gt_img_file)

            #Ratio  
            k = 3
            ratio_img_file = os.path.join(complete_path,"{}_ratio.png".format(i))
            thresh_calc = np.mean(res_noise_deep[0,0])+k*np.std(res_noise_deep[0,0])
            dim = np.clip(res_noise_deep[0,0],0,thresh_calc)
            dim = dim/thresh_calc *255
            dim = Image.fromarray(dim.astype('float64')).convert('L')
            dim.save(ratio_img_file)
            
            #Ratio histogram
            fig = plot_speckle_histogram(res_noise_deep)
            histogram_path = os.path.join(complete_path,"{}_ratio_hist.png".format(i))
            with open(histogram_path,'wb') as f:
                fig.savefig(f, bbox_inches='tight')

    if save_img_flag:
        mean_ratio_filepath = os.path.join(complete_path,"ratio_means.txt")
        with open(mean_ratio_filepath,'w') as f:
            np.savetxt(f,np.array(mean_ratio_list))
        std_ratio_filepath = os.path.join(complete_path,"ratio_stds.txt")
        with open(std_ratio_filepath,'w') as f:
            np.savetxt(f,np.array(std_ratio_list))
            
    print("Test  PSNR Denoised : {}".format(np.mean(psnr_list)))
    print("Test  PSNR Noisy : {}".format(np.mean(psnr_noisy_list)))
    print("Test  Loss {}".format(np.mean(loss_test_list)))
    print("Test  ratio: {}".format(np.mean(mean_ratio_list)))
    print("Test  ratio std: {}".format(np.mean(std_ratio_list)))

    return np.mean(psnr_list), np.mean(psnr_noisy_list), np.mean(loss_test_list), np.mean(mean_ratio_list)
