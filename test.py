import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
import math
from lib.dataset import is_image_file
from PIL import Image
from os import listdir
import os

from model import lowlightnet3 # LOL


def eval(opt):
    device = torch.device('cuda:{}'.format(opt.device))
    model = lowlightnet3()
    model = model.to(device)
    model.load_state_dict(torch.load(opt.modelfile))
    model.eval()
    LL_filename = os.path.join(opt.test_folder)
    est_filename = os.path.join(opt.output)
    try:
        os.stat(est_filename)
    except:
        os.mkdir(est_filename)
    LL_image = [join(LL_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
    print(LL_filename)
    Est_img = [join(est_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
    print(Est_img)
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    time_ave = 0

    for i in range(LL_image.__len__()):
        with torch.no_grad():
            LL_in = Image.open(LL_image[i]).convert('RGB')
            LL = trans(LL_in)
            LL = LL.unsqueeze(0)
            LL = LL.to(device)

            t0 = time.time()
            prediction = model(LL)

            t1 = time.time()
            time_ave += (t1 - t0)
            
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)

            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            Image.fromarray(np.uint8(prediction)).save(Est_img[i])
            print("===> Processing Image: %04d /%04d in %.4f s." % (i, LL_image.__len__(), (t1 - t0)))
    # print("Processing FPS: {} ".format(time_ave / LL_image.__len__() ))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='low-light image enhancement by SMNet')
    parser.add_argument('--test_folder', type=str, default='./datasets/LOL/test/low',help='Location to input images')
    parser.add_argument('--modelfile', default='./model.pth', help='pretrained model')
    parser.add_argument('--output', default='./output_test', help='Location to save output images')
    parser.add_argument('--device', type=str, default='0')

    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=256, help='0 to use original frame size')
    parser.add_argument('--stride', type=int, default=16, help='0 to use original patch size')
    parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    # parser.add_argument('--image_based', type=bool, default=True, help='use image or video based ULN')
    # parser.add_argument('--chop', type=bool, default=False)
    # parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
    # parser.add_argument('--chop_forward', type=bool, default=True)
    
    opt = parser.parse_args()
    gpus_list = range(opt.gpus)
    print(opt)

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    eval(opt)