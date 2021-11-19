import sys
import os
from pathlib import Path
import torch
import numpy as np
from Network.SSDCNet import SSDCNet_classify
from time import time
from PIL import Image
from torchvision import transforms
from load_data_V2 import get_pad
import scipy.io as sio
import cv2

verbose = False
cuda = False


def vprint(*data):
    if verbose:
        print(*data)


FILE_DIR = str(Path(sys.argv[0]).parent) + str(os.sep)
FILE_DIR = os.path.abspath(FILE_DIR)
vprint(FILE_DIR)


def init_ssdc(cuda=True):
    print('loading ssdc')
    mod_path = './model/SHB/best_epoch.pth'
    max_num = 7
    step = 0.5
    label_indice = np.arange(step, max_num+step, step)
    add = np.array([1e-6, 0.05, 0.10, 0.15, 0.20,
                    0.25, 0.30, 0.35, 0.40, 0.45])
    label_indice = np.concatenate((add, label_indice))
    label_indice = torch.Tensor(label_indice)
    class_num = len(label_indice)+1
    div_times = 2
    psize, pstride = 64, 64
    if cuda:
        print('with cuda')
        net = SSDCNet_classify(class_num, label_indice, div_times=div_times,
                               frontend_name='VGG16', block_num=5,
                               IF_pre_bn=False, IF_freeze_bn=False, load_weights=True,
                               psize=psize, pstride=pstride, parse_method='maxp').cuda()
    else:
        print('without cuda')
        net = SSDCNet_classify(class_num, label_indice, div_times=div_times,
                               frontend_name='VGG16', block_num=5,
                               IF_pre_bn=False, IF_freeze_bn=False, load_weights=True,
                               psize=psize, pstride=pstride, parse_method='maxp').cpu()
    if not os.path.isabs(mod_path):
        mod_path = os.path.join(FILE_DIR, mod_path)
    if os.path.exists(mod_path):
        print('model found')
        if cuda:
            all_state_dict = torch.load(mod_path)
        else:
            all_state_dict = torch.load(mod_path, map_location='cpu')
        net.load_state_dict(all_state_dict['net_state_dict'])
        tmp_epoch_num = all_state_dict['tmp_epoch_num']
    else:
        print('os.path.exists(mod_path)=false')
    return net
    


def detect_crowd(img, net, cuda=True):
    # mod_path = './model/SHB/best_epoch.pth'
    # max_num = 7
    # step = 0.5
    # label_indice = np.arange(step, max_num+step, step)
    # add = np.array([1e-6, 0.05, 0.10, 0.15, 0.20,
    #                 0.25, 0.30, 0.35, 0.40, 0.45])
    # label_indice = np.concatenate((add, label_indice))
    # label_indice = torch.Tensor(label_indice)
    # class_num = len(label_indice)+1
    # div_times = 2
    # psize, pstride = 64, 64
    # if cuda:
    #     net = SSDCNet_classify(class_num, label_indice, div_times=div_times,
    #                            frontend_name='VGG16', block_num=5,
    #                            IF_pre_bn=False, IF_freeze_bn=False, load_weights=True,
    #                            psize=psize, pstride=pstride, parse_method='maxp').cuda()
    # else:
    #     net = SSDCNet_classify(class_num, label_indice, div_times=div_times,
    #                            frontend_name='VGG16', block_num=5,
    #                            IF_pre_bn=False, IF_freeze_bn=False, load_weights=True,
    #                            psize=psize, pstride=pstride, parse_method='maxp').cpu()
    # if not os.path.isabs(mod_path):
    #     mod_path = os.path.join(FILE_DIR, mod_path)
    # if os.path.exists(mod_path):
    #     if cuda:
    #         all_state_dict = torch.load(mod_path)
    #     else:
    #         all_state_dict = torch.load(mod_path, map_location='cpu')
    #     net.load_state_dict(all_state_dict['net_state_dict'])
    #     tmp_epoch_num = all_state_dict['tmp_epoch_num']
    # else:
    #     print('os.path.exists(mod_path)=false')
    with torch.no_grad():
        net.eval()
        rgb_dir = './data/SH_partB/rgbstate.mat'
        mat = sio.loadmat(rgb_dir)
        rgb = mat['rgbMean'].reshape(1, 1, 3)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = transforms.ToTensor()(image)
        image = image[None, :, :, :]
        image = get_pad(image, DIV=64)
        image = image - torch.Tensor(rgb).view(3, 1, 1)
        if cuda:
            image = image.cuda()
        image = image.type(torch.float32)
        features = net(image)
        div_res = net.resample(features)
        merge_res = net.parse_merge(div_res)
        outputs = merge_res['div'+str(net.div_times)]
        del merge_res
        pre = int((outputs).sum())
    return pre

if __name__ == '__main__':
    # if len(sys.argv) == 2:
    #     img_path = sys.argv[1]
    #     if not os.path.isabs(img_path):
    #         img_path = os.path.join(FILE_DIR, img_path)
    #     if not os.path.exists(img_path):
    #         print("File not found", file=sys.stderr)
    #         exit()
    # else:
    #     print("Please enter file name", file=sys.stderr)
    #     exit()
    # print('starting')
    # mod_path = './model/SHB/best_epoch.pth' #
    # max_num = 7
    # step = 0.5
    # label_indice = np.arange(step, max_num+step, step)
    # add = np.array([1e-6, 0.05, 0.10, 0.15, 0.20,
    #                 0.25, 0.30, 0.35, 0.40, 0.45])
    # label_indice = np.concatenate((add, label_indice))
    # label_indice = torch.Tensor(label_indice)
    # class_num = len(label_indice)+1
    # div_times = 2
    # psize, pstride = 64, 64
    # if cuda:
    #     print('cuda')
    #     net = SSDCNet_classify(class_num, label_indice, div_times=div_times,
    #                            frontend_name='VGG16', block_num=5,
    #                            IF_pre_bn=False, IF_freeze_bn=False, load_weights=True,
    #                            psize=psize, pstride=pstride, parse_method='maxp').cuda()
    # else:
    #     print('no cuda')
    #     net = SSDCNet_classify(class_num, label_indice, div_times=div_times,
    #                            frontend_name='VGG16', block_num=5,
    #                            IF_pre_bn=False, IF_freeze_bn=False, load_weights=True,
    #                            psize=psize, pstride=pstride, parse_method='maxp').cpu()
    # if not os.path.isabs(mod_path):
    #     mod_path = os.path.join(FILE_DIR, mod_path)
    # if os.path.exists(mod_path):
    #     if cuda:
    #         all_state_dict = torch.load(mod_path)
    #     else:
    #         all_state_dict = torch.load(mod_path, map_location='cpu') #
    #     net.load_state_dict(all_state_dict['net_state_dict'])
    #     tmp_epoch_num = all_state_dict['tmp_epoch_num']
    #     with torch.no_grad():
    #         net.eval()
    #         rgb_dir = './data/SH_partB/rgbstate.mat' #
    #         mat = sio.loadmat(rgb_dir)
    #         rgb = mat['rgbMean'].reshape(1, 1, 3)
    #         image = Image.open(img_path).convert('RGB')
    #         image = transforms.ToTensor()(image)
    #         image = image[None, :, :, :]
    #         image = get_pad(image, DIV=64)
    #         image = image - torch.Tensor(rgb).view(3, 1, 1)
    #         if cuda:
    #             image = image.cuda()
    #         image = image.type(torch.float32)
    #         features = net(image)
    #         div_res = net.resample(features)
    #         merge_res = net.parse_merge(div_res)
    #         outputs = merge_res['div'+str(net.div_times)]
    #         del merge_res
    #         pre = (outputs).sum()
    #         print('%d' % (pre))
    # else:
    #     print('os.path.exists(mod_path)=false')
    ssdc = init_ssdc(True)
    src = cv2.imread('./inference/images/IMG_10.jpg')
    print(detect_crowd(img=src,net=ssdc,cuda=True))
