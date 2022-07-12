'''
Author: sy
Date: 2021-04-09 14:52:20
LastEditors: pida0
LastEditTime: 2022-07-12 16:50:22
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

from .tools import get_train_img_attr, get_test_img_attr, build_transform, pil_loader, get_prt_data_label, get_labels


group_lst_down = {13: [1, 3, 4, 5, 8, 9, 11, 12, 15, 17, 23, 28, 35],
                  6: [7, 19, 27, 29, 30, 34],
                  9: [6, 14, 16, 21, 22, 24, 36, 37, 38],
                  12: [0, 2, 10, 13, 18, 20, 25, 26, 31, 32, 33, 39]}

group_lst_pre = {2: [0, 1],
                 2: [2, 7],
                 2: [3, 4],
                 2: [5, 6]}



class LFWA(data.Dataset):

    def __init__(self, cfg, partition, type, group=[12, 13, 6, 9]):
        self.img_dir = cfg.IMG_DIR
        self.mask_dir = cfg.MASK_DIR
        self.prt_img_dir = cfg.PRT_IMG_DIR
        self.img_augment_dir = cfg.AUGMENT_DIR
        self.group = group
        
        self.type = type
        self.pct_topk = cfg.PCT_TOPK
        self.patch_num = cfg.PATCH_NUM
        self.prt_label = np.random.randint(0,self.patch_num,(20000,1))

        if partition == '0':
            self.attr_file = cfg.TRAIN_FILE
            self.imgs, self.attr = get_train_img_attr(self.attr_file, self.img_dir,
                                                      self.img_augment_dir, cfg.SELECT_RATE)
            
        else:
            self.attr_file = cfg.TEST_FILE
            self.imgs, self.attr = get_test_img_attr(self.attr_file, self.img_dir, 1)

        self.trans = build_transform(self.type)
        self.length = len(self.imgs)

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
            
        elif self.type.find('downstream')!=-1:
            img = self.trans(full_img)
            if self.group != None:
                attrs = [0] * 4
                for i in range(len(self.group)):
                    attrs[i] = self.attr[index, group_lst_down[self.group[i]]]

                return img, attrs
            else:
                return img, self.attr[index,:]
        else:
            raise ValueError('type {} not supported. Only support pretext-train、pretext-val、downstream-train、downstream-val'.format(self.type))
        

    def __len__(self):
        return self.length