import os.path
import logging
import torchvision.transforms.functional as F
import time
from collections import OrderedDict
import torch
import cv2
import torch.nn as nn
import numpy as np
from utils import utils_logger
#from utils import utils_image as util
from RFDN import RFDN
from datasets import load_dataset
from torchvision import transforms
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob

class ImagePairDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None,transform_out=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_files = os.listdir(input_dir)
        self.target_files = os.listdir(target_dir)
        self.transform = transform
        self.transform_out = transform_out
    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, self.input_files[index])
        target_path = os.path.join(self.target_dir, self.target_files[index])
        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')

        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform_out(target_image)

        return input_image, target_image

    def __len__(self):
        return min(len(self.input_files), len(self.target_files))
    
def rgb_to_yuv444(rgb_tensor):
    # Reshape the tensor to separate channels
    r = rgb_tensor[:, 0,:,:]
    g = rgb_tensor[:, 1,:,:]
    b = rgb_tensor[:, 2,:,:]

    # Y component calculation
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # U component calculation
    u = r - y

    # V component calculation
    v = b - y

    # Stack the Y, U, and V components along the channel dimension
    yuv_tensor = torch.stack([y, u, v], dim=1)

    return yuv_tensor

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    max_pixel = 1.0
    psnr = 10 * np.log10(255 / np.sqrt(mse))
    return psnr

def main():

    utils_logger.logger_info('AIM-track', log_path='AIM-track.log')
    logger = logging.getLogger('AIM-track')

    # --------------------------------
    # basic settings
    # --------------------------------
    
    #torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # load model
    # --------------------------------
    model_path = os.path.join('trained_model', 'RFDN_AIM.pth')
    model = RFDN()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # --------------------------------
    # read image
    # --------------------------------
    input_dir = 'div2k/LR'
    target_dir = 'div2k/HR'
    transform = transforms.Compose([transforms.Resize(((620,1020))),
                                    transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_out = transforms.Compose([transforms.Resize((1240,2040)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    
    dataset = ImagePairDataset(input_dir, target_dir, transform=transform,transform_out=transform_out)


    valid_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)

    # L_folder = os.path.join(testsets, testset_L, 'X4')
    # E_folder = os.path.join(testsets, testset_L+'_results')
    # util.mkdir(E_folder)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    # logger.info(L_folder)
    # logger.info(E_folder)
    model.eval()
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    img_SR = []
    valid_loss = 0
    criterion = nn.MSELoss()
    # loop over the validation data and calculate the PSNR values
    with torch.no_grad():
        psnr_y_total = 0
        psnr_u_total = 0
        psnr_v_total = 0
        count = 0

        for input,target in valid_loader:
            # get the input and target images
            input_img = input.to(device)
            target_img = target
            
            target_img_yuv = rgb_to_yuv444(target_img)
            target = target.to(device)
            # generate the super-resolved image using the model
            start.record()
            output_img_rgb = model(input_img)
            end.record()
             # Add a batch dimension to the tensor
            img_tensor = output_img_rgb

            # Downscale the image by a factor of 2
            downscaled_tensor = torch.nn.functional.interpolate(img_tensor, scale_factor=0.5, mode='bilinear', align_corners=True)

            # Remove the batch dimension from the tensor
            output_img_rgb = downscaled_tensor

            loss = criterion(output_img_rgb, target)
            valid_loss += loss.item()
            torch.cuda.synchronize()
            test_results['runtime'].append(start.elapsed_time(end))  # milliseconds
            img_SR.append(output_img_rgb)

           
            output_img_yuv = rgb_to_yuv444(output_img_rgb)


            output_img_y = output_img_yuv[:, 0, :, :]
            output_img_u = output_img_yuv[:, 1, :, :]
            output_img_v = output_img_yuv[:, 2, :, :]

            # calculate the PSNR for the Y, U, and V channels
            psnr_y = calculate_psnr(output_img_y.cpu().numpy().squeeze(), target_img_yuv[:, 0, :, :].cpu().numpy().squeeze())
            psnr_u = calculate_psnr(output_img_u.cpu().numpy().squeeze(), target_img_yuv[:, 1, :, :].cpu().numpy().squeeze())
            psnr_v = calculate_psnr(output_img_v.cpu().numpy().squeeze(), target_img_yuv[:, 2, :, :].cpu().numpy().squeeze())
            logger.info("psnr y: {} , psnr u: {} , psnr v: {}".format(psnr_y,psnr_u,psnr_v))

            # add the PSNR values to the total and update the count
            psnr_y_total += psnr_y
            psnr_u_total += psnr_u
            psnr_v_total += psnr_v
            count += 1

        valid_loss /= len(valid_loader.dataset)
        # calculate the average PSNR values
        avg_psnr_y = psnr_y_total / count
        avg_psnr_u = psnr_u_total / count
        avg_psnr_v = psnr_v_total / count
        logger.info("valid loss:{}, avg psnr y: {} , avg psnr u: {} , avg psnr v: {}".format(valid_loss,avg_psnr_y,avg_psnr_u,avg_psnr_v))
        # print the results
        print("Average PSNR (Y): {:.2f} dB".format(avg_psnr_y))
        print("Average PSNR (U): {:.2f} dB".format(avg_psnr_u))
        print("Average PSNR (V): {:.2f} dB".format(avg_psnr_v))

    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime is : {:.6f} seconds'.format(ave_runtime))

   
if __name__ == '__main__':

    main()
