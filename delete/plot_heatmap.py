'''
Author: sy
Date: 2021-04-06 09:18:56
LastEditors: your name
LastEditTime: 2021-10-13 16:17:11
Description: 用来画热力图
'''

#coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim import lr_scheduler
import datetime
from tensorboardX import SummaryWriter as sw
import cv2
import numpy as np

from utils import cal_attr_acc, lambda_warm_up, lambda_warm_up_2, lambda_warm_up_sgd
from configs import cfg, show_cfg
from datasets import make_dataloader
from models import get_down_model, get_down_model_5b

part={0:'whole',1:'upper',2:'middle',3:'lower'}

def plot_map(gpu_id, cfg_pth):
    parser = argparse.ArgumentParser(description="S3NET  Downstream Training")
    parser.add_argument("--config_file", default=cfg_pth, help="path to config file", type=str)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    

    #======================= load data and model =======================
    trainloader, valloader, testloader = make_dataloader(cfg, group=[12, 13, 6, 9])

    model = get_down_model_5b(num_classes_cls=[12, 13, 6, 9])
    
    if cfg.TRAIN.IS_CONTINUE:
        # model_path = cfg.TRAIN.LOAD_PTH + 'epoch{}.pth'.format(cfg.TRAIN.LOAD_EPOCH)
        model_path = cfg.TRAIN.LOAD_PTH
        # model.load_state_dict(torch.load(model_path))
        model.load_param_multi_gpu(model_path)
    elif cfg.TRAIN.IS_PRETRAINED:
        model_path = cfg.TRAIN.PRETRAIN_PTH + 'epoch{}.pth'.format(cfg.TRAIN.PRETRAIN_EPOCH)
        model.load_param(model_path)
    else:
        model_path = 'baseline'
    

    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

    #======================= set loss and optimizer ====================
    criterion = nn.BCEWithLogitsLoss() # 还没测试
    # criterion = BalancedLoss()

    # optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
    #                         lr=cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=5e-4)
                            
    # scheduler=lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warm_up_2)
    # scheduler=lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5,last_epoch=-1)
    # scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
    #                                          patience=10, verbose=True)

    #====================== show info =================================
    show_cfg(cfg)
    print('\nusing dataset [{}], select rate: {}'.format(cfg.DATA.DATASET_NAME,cfg.DATA.SELECT_RATE))  
    print('trainloader: ',len(trainloader))  
    print('testloader: ',len(testloader))
    print('Model loaded from {}......'.format(model_path))

    #====================== train start ===============================
    iter_counter=0
    
    correct = torch.FloatTensor(40).fill_(0)
    total = 0

    for batch_idx, (imgs, attrs, img_pth) in enumerate(trainloader):
        if batch_idx == 10:
            break 
        
        images = imgs.cuda()
        for i in range(4):
            attrs[i] = attrs[i].cuda().type(torch.cuda.FloatTensor)
        output, masks = model(images)
            # print(output)
            # print(attrs)

        loss = [0] * 4
        total_loss = 0
        for i in range(4):
            loss[i] = criterion(output[i], attrs[i])
            total_loss += loss[i]
                


        pred = torch.cat(output, 1)
        label = torch.cat(attrs, 1)
        c, t = cal_attr_acc(pred.cpu().data, label.cpu().data, 40)
        correct.add_(c)
        total += t
        print('| current accuracy per attr:\n',correct / total)
        mean_acc = torch.mean(correct / total)
        print('| current mean accuracy: ',mean_acc)

        
        

        #=========================== plot heatmap ============================
        print('masks size: ',len(masks))
        for idm, mask_idm in enumerate(masks):
            # mask_idm 1*512*7*7
            for i, mask in enumerate(mask_idm):
                # mask 512*7*7
                pth=img_pth[i]
                img=cv2.imread(pth)

                mask = torch.sum(mask, dim=0) #按通道维度相加
                mask = torch.unsqueeze(mask, 2) # 7*7*1
                mask = mask.cpu().data.numpy()

                mask = cv2.resize(mask, (img.shape[1], img.shape[0])) #cv2: w,h,c
                mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
                mask = np.uint8(255 * mask) 
                mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                superimposed_img = img * 0.5 + mask * 0.5

                if not os.path.exists(cfg.PLOT.SAVE_ROOT):
                    os.mkdir(cfg.PLOT.SAVE_ROOT)
                idx=pth.split('/')[-1].split('.')[0]
                save_path=cfg.PLOT.SAVE_ROOT+'/'+idx
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path=save_path+'/'+part[idm]+'_test.png'
                print('Saving to', save_path)
                cv2.imwrite(save_path, superimposed_img)

if __name__ == '__main__':
    cfg_pth='./config_files/train_downstream.yml'
    plot_map('0', cfg_pth)
