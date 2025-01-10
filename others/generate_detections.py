"""
@Author: Du Yunhao
@Filename: generate_detections.py
@Contact: dyh_bupt@163.com
@Time: 2021/11/8 17:02
@Discription: 生成检测特征
"""
import os
import cv2
import sys
import glob
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from torchvision import transforms
from os.path import join, exists, split
from model import Net
import torch.backends.cudnn as cudnn
import torchvision.transforms as tt

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from vit_pytorch import ViT
import timm

sys.path.append('.')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available(): 
    cudnn.benchmark = True

def get_model():
    model = Net()
    # model.load_param()
    return model

def get_transform(size=(280, 140)):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        norm,
    ])
    return transform

if __name__ == '__main__':
    print(datetime.now())
    '''配置信息'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # cfg.MODEL.BACKBONE.WITH_IBN = False

    thres_score = 0.6
    root_img = '/mnt/data/kaiwen/code/deepsort/StrongSORT/MOT17/train'
    # root_img = 'MOT17/test'
    # root_img = '/data/dyh/data/MOTChallenge/MOT20/test'
    # dir_in_det = '/data/dyh/results/StrongSORT/Detection/YOLOX_ablation_nms.8_score.1'
    # dir_in_det = '/data/dyh/results/StrongSORT/TEST/MOT17_YOLOX_nms.8_score.1'
    dir_in_det = '/mnt/data/kaiwen/code/deepsort/StrongSORT/detections_mot17'
    # dir_out_det = '/data/dyh/results/StrongSORT/Features/YOLOX_nms.8_score.6_BoT-S50_DukeMTMC_again'
    # dir_out_det = '/data/dyh/results/StrongSORT/TEST/MOT17_YOLOX_nms.8_score.1_BoT-S50'
    dir_out_det = '/mnt/data/kaiwen/code/deepsort/StrongSORT/features_mot17_vitg'
    if not exists(dir_out_det): os.mkdir(dir_out_det)
    model = get_model()

    # model = torch.hub.load(r'/home/kaiwen/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitg14', source='local').cuda()
    model_weights = '/mnt/data/kaiwen/code/deepsort/StrongSORT/others/ckpt_g.t7'
    # model_weights = '/mnt/data/kaiwen/code/deepsort/StrongSORT/others/original_ckpt.t7'
    state_dict = torch.load(model_weights, map_location=lambda storage, loc: storage)['net_dict']
    # state_dict = torch.load(model_weights, map_location=lambda storage, loc: storage)
    
    # device_ids = [0, 1]
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict)
    model.to(device)

    transform = get_transform((448, 224))
    # transform = get_transform((256, 128))
    # transform = get_transform((384, 128))
    # transform = get_transform((128, 64))

    files = sorted(glob.glob(join(dir_in_det, '*.txt')))
    for i, file in enumerate(files, start=1):
        # if i <= 5: continue
        video = split(file)[1][:-4]
        print('processing the video {}...'.format(video))
        dir_img = join(root_img, '{}/img1'.format(video))
        detections = np.loadtxt(file, delimiter=',') # MOT17 MOT20
        # detections = np.loadtxt(file, delimiter=' ') # MOT16
        detections = detections[detections[:, 6] >= thres_score]
        mim_frame, max_frame = int(min(detections[:, 0])), int(max(detections[:, 0]))
        list_res = list()
        for frame in range(mim_frame, max_frame + 1):
            # print('  processing the frame {}...'.format(frame))
            img = Image.open(join(dir_img, '%06d.jpg' % frame))
            detections_frame = detections[detections[:, 0] == frame]
            batch = [img.crop((b[2], b[3], b[2] + b[4], b[3] + b[5])) for b in detections_frame]
            batch = [transform(patch) * 255. for patch in batch]
            if batch:
                batch = torch.stack(batch, dim=0).cuda()
                # input_tensor = torch.Tensor(batch.cpu().numpy())
                # input_tensor = input_tensor.cuda()
                # with torch.no_grad():
                #     result = model.backbone.forward_features(input_tensor)
                # patch_tokens = result['x_norm_patchtokens'].cpu().detach().numpy().reshape([batch.shape[0],256,-1])
                # fg_pca = PCA(n_components=1)
                # masks=[]
                # all_patches = patch_tokens.reshape([-1,768])
                # reduced_patches = fg_pca.fit_transform(all_patches)
                # norm_patches = minmax_scale(reduced_patches)
                # image_norm_patches = norm_patches.reshape([batch.shape[0],256])
                # for i in range(batch.shape[0]):
                #     image_patches = image_norm_patches[i,:]
                #     mask = (image_patches > 0.6).ravel()
                #     masks.append(mask)
                #     image_patches[np.logical_not(mask)] = 0
                # object_pca = PCA(n_components=3)
                # mask_indices = [0, *np.cumsum([np.sum(m) for m in masks]), -1]
                # fg_patches = np.vstack([patch_tokens[i,masks[i],:] for i in range(batch.shape[0])])
                # reduced_patches = object_pca.fit_transform(fg_patches)
                # reduced_patches = minmax_scale(reduced_patches)
                # outputs = []
                # for i in range(batch.shape[0]):
                #     patch_image = np.zeros((256,3), dtype='float32')
                #     patch_image[masks[i],:] = reduced_patches[mask_indices[i]:mask_indices[i+1],:]
                #     outputs.append(patch_image.reshape([256 * 3]))
                # outputs = np.array(outputs)
                # result = model.forward_features(batch)
                # outputs = result['x_norm_clstoken'].detach().cpu().numpy()
                outputs = model(batch).cpu().detach().numpy()
                # outputs = model(batch)[0][1].detach().cpu().numpy()
                list_res.append(np.c_[(detections_frame, outputs)])
        res = np.concatenate(list_res, axis=0)
        np.save(join(dir_out_det, video + '.npy'), res, allow_pickle=False)
    print(datetime.now())



