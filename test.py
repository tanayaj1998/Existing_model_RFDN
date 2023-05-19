import os.path
import logging
import time
from collections import OrderedDict
import torch
import argparse
from utils import utils_logger
from utils import utils_image as util
from RFDN import RFDN
import cv2
import sys
import numpy as np
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--test_data_lr', type=str, default='LR/Bosphorus_1920x1080_120F.yuv')
    parser.add_argument('--test_data_hr', type=str, default='data/Bosphorus_3840x2160_120F.yuv')
    parser.add_argument('--patch_size', type=str, default='256x256')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--model_path', type=str, default="checkpoint2/model.pt")
    parser.add_argument('--interpolate', type=str, default="bilinear")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    log_name = ""
    if args.interpolate == "bilinear":
        log_name+="testing_bilinear_logs"
    else:
        log_name+="testing_bicubic_logs"
    utils_logger.logger_info(log_name, log_path=log_name+'.log')
    logger = logging.getLogger(log_name)

    # --------------------------------
    # basic settings
    # --------------------------------
    

    torch.cuda.current_device()
    torch.cuda.empty_cache()
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

    width, height =1920,1080
    width_hr, height_hr = 3840,2160
    # Number of frames: in YUV420 frame size in bytes is width*height*1.5
    n_frames = 120
    print("frames:",n_frames)
    # Open 'input.yuv' a binary file.
    f_lr = open(args.test_data_lr, 'rb')
    f_hr =open(args.test_data_hr,'rb')

    
    frame_size_lr = int(width * height * 1.5)
    frame_size_hr = int(width_hr * height_hr * 1.5)
    i=0
    SR = []
    PSNR_Y = []
    PSNR_U = []
    PSNR_V = []
    while True:
        # Read a single YUV420 frame
        yuv420_frame_lr = f_lr.read(frame_size_lr)
        yuv420_frame_hr = f_hr.read(frame_size_hr)
        # If we've reached the end of the video file, break out of the loop
        if not( yuv420_frame_lr and yuv420_frame_hr):
            break
        # Reshape the YUV420 frame into a 3D numpy array
        yuv420_frame_lr = np.frombuffer(yuv420_frame_lr, dtype=np.uint8).reshape((int(height * 1.5), width))
        rgb_lr = cv2.cvtColor(yuv420_frame_lr, cv2.COLOR_YUV2BGR_I420)
        cv2.imwrite('output/LR/'+str(i)+'.png',rgb_lr)
        print(np.frombuffer(yuv420_frame_hr, dtype=np.uint8).shape)
        yuv420_frame_hr = np.frombuffer(yuv420_frame_hr, dtype=np.uint8).reshape((int(height_hr * 1.5), width_hr))
        rgb_hr = cv2.cvtColor(yuv420_frame_hr, cv2.COLOR_YUV2BGR_I420)

        cv2.imwrite('output/HR/'+str(i)+'.png',rgb_hr)
        #print(yuv420_frame_lr.shape)
        # Extract the Y, U, and V components from the YUV420 frame
        y =( yuv420_frame_lr[:height, :]).reshape((height, width))
        u = ( yuv420_frame_lr[height:int(height*1.25), :]).reshape((height//2, width//2))
        v = (yuv420_frame_lr[int(height*1.25):, :]).reshape((height//2, width//2))

    
        # Upsample the U and V components to YUV444 (bilinear) for bicubic INTER_CUBIC
        if args.interpolate =="bilinear":
            u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
        elif args.interpolate =="bicubic":
            u = cv2.resize(u, (width, height), interpolation=cv2.INTER_CUBIC)
            v = cv2.resize(v, (width, height), interpolation=cv2.INTER_CUBIC)
        

        yuv = np.dstack((y,u,v))
        #print(yuv.shape)
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)/255.
        #print(yuv.shape)
        #bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        #cv2.imwrite('test.png',bgr)
        rgb= torch.from_numpy(rgb).unsqueeze(0).float()
        # u=torch.from_numpy(u).unsqueeze(0).float()
        # v =torch.from_numpy(v).unsqueeze(0).float()

       
        #Combine the Y, U, and V components into a YUV444 frame
        rgb = rgb.permute(0,3, 1, 2).float().to(device)
        
        #yuv444=yuv444/255.
        #yuv = torch.from_numpy(yuv).unsqueeze(0).permute(0,3,1,2).float().to(device)
        #print(yuv.shape)
        output_rgb = model(rgb)
        
        output_rgb = output_rgb.squeeze(0).cpu().numpy().transpose(1,2,0)
        output = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2YUV)
        
        y= output[:,:,0]
        u= output[:,:,1]
        v=output[:,:,2]
        y = cv2.resize(y, (y.shape[1]//2, y.shape[0]//2), interpolation=cv2.INTER_AREA)
        u = cv2.resize(u, (u.shape[1]//2, u.shape[0]//2), interpolation=cv2.INTER_AREA)
        v = cv2.resize(v, (v.shape[1]//2, v.shape[0]//2), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(np.dstack((y,u,v)), cv2.COLOR_YUV2BGR)*255  
        #rgb = cv2.cvtColor(output_444, cv2.COLOR_YUV2RGB_I420)
        cv2.imwrite('output/SR/'+str(i)+'.png',bgr)
        i+=1

        #converting y,u and v from yuv444 to yuv420
        #downsample u and v
        y_sr = y
        u_sr = cv2.resize(u, (width_hr//2, height_hr//2), interpolation=cv2.INTER_AREA)
        v_sr = cv2.resize(v, (width_hr//2, height_hr//2), interpolation=cv2.INTER_AREA) 

        y_hr =( yuv420_frame_hr[:height_hr, :]).reshape((height_hr, width_hr))/255.
        u_hr = ( yuv420_frame_hr[height_hr:int(height_hr*1.25), :]).reshape((height_hr//2, width_hr//2))/255.
        v_hr = (yuv420_frame_hr[int(height_hr*1.25):, :]).reshape((height_hr//2, width_hr//2))/255.

        mse_y = np.mean((y_hr - y_sr) ** 2)
        mse_u = np.mean((u_hr - u_sr) ** 2)
        mse_v = np.mean((v_hr - v_sr) ** 2)

        psnr_y=  20 * np.log10(1/ np.sqrt(mse_y))
        psnr_u= 20 * np.log10(1/ np.sqrt(mse_u))
        psnr_v= 20 * np.log10(1/ np.sqrt(mse_v))
        

        logger.info("psnr y: {} , psnr u: {} , psnr v: {}".format(psnr_y,psnr_u,psnr_v))
        PSNR_Y.append(psnr_y)
        PSNR_U.append(psnr_u)
        PSNR_V.append(psnr_v)

    f_lr.close()
    f_hr.close()
    avg_psnr_y =sum(PSNR_Y)/len(PSNR_Y)
    avg_psnr_u =sum(PSNR_U)/len(PSNR_U)
    avg_psnr_v =sum(PSNR_V)/len(PSNR_V)
    logger.info("avg psnr y: {} , avg psnr u: {} ,avg psnr v: {}".format(avg_psnr_y,avg_psnr_u,avg_psnr_v))
        # --------------------------------
        # (3) save results
        # --------------------------------
        #util.imsave(img_E, os.path.join(E_folder, img_name+ext))

    #ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    #logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))

    # --------------------------------
    # (4) calculate psnr
    # --------------------------------
    '''
    psnr = []
    idx = 0
    H_folder = '/home/lj/EfficientSR-1.5.0/train/dataset/benchmark/DIV2K_valid/HR/'
    for img in util.get_image_paths(H_folder):
        img_H = util.imread_uint(img, n_channels=3)
        psnr.append(util.calculate_psnr(img_SR[idx], img_H))
        idx += 1
    logger.info('------> Average psnr of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr)/len(psnr)))
    '''

if __name__ == '__main__':

    main(sys.argv[1:])
