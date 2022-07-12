'''
Author: sy
Date: 2021-04-01 08:58:47
LastEditors: pida0
LastEditTime: 2022-07-12 16:52:17
Description: file content
'''

#coding:utf-8

from PIL import Image
import torch.utils.data as data
import numpy as np
import os

import torchvision.transforms as T
import torch
from torchvision.utils import make_grid,save_image
import matplotlib.pyplot as plt




from .tools import *

group_lst_down = {16: [9, 11, 12, 14, 15, 16, 17, 28, 29, 30, 31, 32, 33, 40, 44, 45],
                  6: [7, 13, 26, 37, 38, 41],
                  8: [18, 19, 21, 25, 34, 36, 42, 43],
                  17: [0, 1, 2, 3, 4, 5, 6, 8, 10, 20, 22, 23, 24, 27, 35, 39, 46]}

group_lst_pre = {2: [0, 1],
                 7: [2, 3,4,5,6,17,18],
                 4: [7,8,9,10],
                 6: [11,12,13,14,15,16]}

class MAAD(data.Dataset):
    def __init__(self, cfg, partition, type, group=[17, 16, 6, 8]):
        self.group = group
        self.patch_num = cfg.PATCH_NUM
        self.prt_label = np.random.randint(0, self.patch_num, (202600, 1))
        self.type = type
        self.pct_topk = cfg.PCT_TOPK  # for pct cls
        self.trans = build_transform(self.type)
        
        
        if partition == "0":
            self.attr_file = np.load(cfg.TRAIN_FILE, allow_pickle=True)
            self.attr = (np.array(self.attr_file[:, 2:], dtype=int) + 1)/2 #-1,0,1-->0,1,2
            
            self.imglist = self.attr_file[:, 0]
            self.attr = self.attr[::cfg.SELECT_RATE]
            self.imglist = self.imglist[::cfg.SELECT_RATE]
            self.img_dir = cfg.TRAIN_IMG_DIR
            self.prt_img_dir = cfg.TRAIN_PRT_IMG_DIR
            self.mask_dir = cfg.TRAIN_MASK_DIR  # for pst
        else:
            self.attr_file = np.load(cfg.TEST_FILE, allow_pickle=True)
            self.attr = (np.array(self.attr_file[:, 2:], dtype=int) + 1)/2 #-1,0,1-->0,1,2

            self.imglist = self.attr_file[:, 0]
            self.img_dir = cfg.TEST_IMG_DIR
            self.prt_img_dir = cfg.TEST_PRT_IMG_DIR
            self.mask_dir = cfg.TEST_MASK_DIR  # for pst

        
        self.length = len(self.imglist)
        

    def __getitem__(self, index):

        full_img = pil_loader(os.path.join(self.img_dir, self.imglist[index]), 'img') #完整的图像

        if self.type.find('pretext') != -1:
            # 完整的语义标签
            full_mask = pil_loader(os.path.join(self.mask_dir, self.imglist[index].split('.')[0] + '.png'), 'mask')
            
            # 对原图和语义标签做相同位置的随机剪裁
            img_trans=T.Compose([T.ToTensor(),T.RandomCrop(224),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            label_trans=T.Compose([T.ToTensor(),T.RandomCrop(224)])
            seed=torch.random.seed()
            torch.random.manual_seed(seed)
            psct_img=img_trans(full_img)
            torch.random.manual_seed(seed)
            pst_label = np.array(full_mask).astype(np.int64)
            pst_label=label_trans(pst_label).squeeze(0)


            pct_label = get_labels(pst_label, self.pct_topk)
            pct_labels = [0] * 4
            for i in range(len(self.group)):
                pct_labels[i] = pct_label[group_lst_pre[self.group[i]]]
            
            prt_img_ = pil_loader(os.path.join(self.prt_img_dir, self.imgs[index]), 'img')
            prt_img_g, prt_label, prt_angle_label= get_prt_data_label(prt_img_, full_mask, self.patch_num)
            
            return [psct_img, prt_img_g], [pst_label, pct_labels, prt_label, prt_angle_label]
            
        elif self.type.find('downstream') != -1:
            img = self.trans(full_img)
            if self.group != None:
                attrs = [0] * 4
                for i in range(len(self.group)):
                    attrs[i] = self.attr[index, group_lst_down[self.group[i]]]
                # attrs = np.array(attrs) 
                return img, attrs
            else:
                return img, self.attr[index,:]

    def __len__(self):
        return self.length

            
if __name__ == '__main__':
    

    cfg_pth='/home/sy/s3net/config_files/test.yml'


            



        
        
            
