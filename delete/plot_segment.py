# '''
# Author: sy
# Date: 2021-10-13 15:42:55
# LastEditors: your name
# LastEditTime: 2021-10-14 09:30:41
# Description: 用来画分割的图
# '''



# #!/usr/bin/python
# # -*- encoding: utf-8 -*-


# from torchvision.transforms.transforms import RandomCrop, RandomResizedCrop
# # from logger import setup_logger
# from models import get_model_5b

# import torch

# import os
# import os.path as osp
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# import cv2

# def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
#     # Colors for all 20 parts
#     # part_colors = [[255, 0, 0],#red
#     #                [0, 255, 170],#green
#     #                [44,18,239], #blue
#     #                [170,0,255],#purple
#     #                [83,83,50],#zong se
#     #                [255, 255, 20],#yellow
#     #                [35,20,57],#black
#     #                [225,85,1]#orange
#     #               ]
#     # part_colors = [[255, 0, 0], [255, 85, 0], #[255, 170, 0],
#     #                #[255, 0, 85], [255, 0, 170],
#     #                #[0, 255, 0], [85, 255, 0], [170, 255, 0],
#     #                #[0, 255, 85], [0, 255, 170],
#     #                [0, 0, 255], [85, 0, 255], [170, 0, 255],
#     #                #[0, 85, 255], [0, 170, 255],
#     #                [255, 255, 0], [255, 255, 85], [255, 255, 170],
#     #                [255, 0, 255], [255, 85, 255], [255, 170, 255],
#     #                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
#     part_colors = [[255, 0, 0], [255, 85, 0], #[255, 170, 0],
#                    #[255, 0, 85], [255, 0, 170],
#                    [0, 255, 0], [85, 255, 0], [170, 255, 0],
#                    [0, 255, 85], [0, 255, 170],
#                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
#                    #[0, 85, 255], [0, 170, 255],
#                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
#                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
#                    [0, 255, 255], [85, 255, 255], [170, 255, 255]]
#     #上面的是rgb排列，真正显示的颜色应该是转成bgr之后的
#     #背景、'skin-1', ('l_brow-2', 'r_brow-3', 'l_eye-4', 'r_eye-5',
#     # 'eye_g-6'), ('l_ear-7', 'r_ear-8', 'ear_r-9'),
#     # 'nose-10', ('mouth-11', 'u_lip-12', 'l_lip-13'), ('neck-14',
#     # 'neck_l-15', 'cloth-16'), ('hair-17', 'hat-18')
#     im = np.array(im)
#     vis_im = im.copy().astype(np.uint8)
#     # print(vis_im.shape)
#     vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
#     # print(vis_parsing_anno.shape)
#     vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
#     # print(vis_parsing_anno.shape)

#     vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

#     num_of_class = np.max(vis_parsing_anno)

#     # 处理skin，其余部分都变成黑色，skin保留原色
#     for pi in range(1, num_of_class + 1):
#         # print(pi)
#         index = np.where(vis_parsing_anno == pi)
#         vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
   
#     # pi=1
#     # index=np.where((vis_parsing_anno>6)|(vis_parsing_anno<2))
#     # print(index)
    
#     # vis_im[index[0],index[1],:]=[0,0,0]

#     vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
#     # print(vis_parsing_anno_color.shape, vis_im.shape)
#     vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
#     # vis_im=cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)

#     # Save result or not
#     if save_im:
#         # cv2.imwrite(save_path[:-4] +'_vpa.png', vis_parsing_anno)
#         # cv2.imwrite(save_path[:-4] +'_pa.png', parsing_anno)
#         cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#     # return vis_im

# def evaluate(savepth='./res/test_res', imgpth='./data', ckp='model_final_diss.pth'):

#     if not os.path.exists(savepth):
#         os.makedirs(savepth)

    
#     net = get_model_5b(num_classes_rot=4,
#                       num_classes_seg=19,
#                       num_classes_cls=[2,7,4,6])
#     net.cuda()

#     net.load_state_dict(torch.load(ckp))
#     net.eval()

#     to_tensor = transforms.Compose([
#         # transforms.RandomResizedCrop(512),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     num=0

#     with torch.no_grad():
#         for image_path in os.listdir(imgpth):
#             num+=1

#             image = Image.open(osp.join(dspth, image_path))
#             image = image.resize((224, 224), Image.BILINEAR)
#             if image.size[0]!=224:
#                 raise ValueError('img size != 224')
            
#             img = to_tensor(image)
#             pr_img=torch.stack((img,img,img,img),0)
#             pr_img=torch.unsqueeze(pr_img,0)
#             pr_img=pr_img.cuda()
#             img = torch.unsqueeze(img, 0)
#             img = img.cuda()

#             _,out,_,_,_,_ = net(pr_img,img)
#             # print(out.shape)
#             parsing = out.squeeze(0).cpu().numpy().argmax(0)
#             # print(parsing)
#             # print(parsing.shape)
#             # print(image_path)
#             # print(np.unique(parsing))
#             if not os.path.exists(savepth):
#                 os.makedirs(savepth)
#             save_path=osp.join(savepth, image_path)
            
#             # print(parsing)
#             # cv2.imwrite(save_path[:-4] +'.png', parsing)
#             if num%500==0:
#                 print(num)
            
#             vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=save_path)
#             # break
#         print("total mask: ",num)





# # -*- coding: utf-8 -*-

# from PIL import Image



# if __name__ == "__main__":
    
# #     filename = r'/media/sy/disk/DataSet/CelebAMask-HQ/test-img/000001.jpg'
# #     img = Image.open(filename)
# #     size = img.size
# #     print(size)
    
# # # 准备将图片切割成9张小图片
# #     weight = int(size[0])
# #     height = int(size[1] // 2)
# # # 切割后的小图的宽度和高度
# #     print(weight, height)
# #     box = (0, 0, weight, height)
# #     img.crop(box).save('up.png')
# #     box = (0, height, weight, height*2)
# #     img.crop(box).save('below.png')
# #     for i in range(10):
# #         # crop=transforms.RandomResizedCrop(512)(img)
# #         crop=transforms.RandomCrop(75)(img)
# #         # crop=transforms.Resize(512)(transforms.RandomCrop(75)(img))
        
# #         crop.save('{}.png'.format(i+10))
#     # for j in range(1):
#     #     for i in range(2):
#     #         box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
#     #         region = img.crop(box)
#     #         region.save('{}{}.png'.format(j, i))

#     #=================== crop the celeba img to 75x75, then resize to 512x512 ================
#     # dspth='/media/sy/disk/DataSet/CelebA/img_align_celeba'
#     # respth='/media/sy/disk/DataSet/CelebA/img_align_celeba_crop75'
#     # savepth='/media/sy/disk/DataSet/CelebA/crop75_mask'

#     # dspth='/media/sy/disk/DataSet/CelebAMask-HQ/test-img'
#     # respth='/media/sy/disk/DataSet/CelebA/test-img_crop75'
#     # savepth='/media/sy/disk/DataSet/CelebA/test-img_crop75_nosmooth'

#     dspth='/media/sy/disk/DataSet/CelebAMask-HQ/test-img'
#     imgpth='/media/data/sy_data/DataSet/LFW/lfw_crop75'
#     savepth='/home/sy/segment_pic/lfw_no_smooth_19'

#     # if not os.path.exists(respth):
#     #     os.makedirs(respth)
#     # num=0
#     # for image_path in os.listdir(dspth):
            
#     #         num+=1
#     #         img = Image.open(osp.join(dspth, image_path))
#     #         crop=transforms.Resize(224)(transforms.RandomCrop(75)(img))
#     #         crop.save(osp.join(respth, image_path))
#     #         if num%100==0:
#     #             print(num)
#     #         # break
#     # print("total pics: ",num)       
#     evaluate(savepth, imgpth=imgpth, ckp='/media/sy/disk/ckp/seg_ssl_v5_pr_celeba_nosmooth/pami/epoch17.pth')



'''
Descripttion: main entry for preprocess images in CelebA for self-supervised learning
version: 1.0
Author: Shu Ying
Date: 2020-08-12 09:30:57
LastEditors: your name
LastEditTime: 2021-10-13 11:14:24
'''

#!/usr/bin/python
# -*- encoding: utf-8 -*-


from torch import randint
from torch.nn import functional as F
from torchvision.transforms.transforms import RandomCrop, RandomResizedCrop
# from logger import setup_logger
# from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import pandas as pd
from models import get_model_5b

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], #[255, 170, 0],
                   #[255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   #[0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    #上面的是rgb排列，真正显示的颜色应该是转成bgr之后的
    #背景、'skin-1', ('l_brow-2', 'r_brow-3', 'l_eye-4', 'r_eye-5',
    # 'eye_g-6'), ('l_ear-7', 'r_ear-8', 'ear_r-9'),
    # 'nose-10', ('mouth-11', 'u_lip-12', 'l_lip-13'), ('neck-14',
    # 'neck_l-15', 'cloth-16'), ('hair-17', 'hat-18')
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    # print(vis_im.shape)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    # print(vis_parsing_anno.shape)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    # print(vis_parsing_anno.shape)

    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    # 处理skin，其余部分都变成黑色，skin保留原色
    for pi in range(1, num_of_class + 1):
        # print(pi)
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
   
    # pi=1
    # index=np.where((vis_parsing_anno>6)|(vis_parsing_anno<2))
    # print(index)
    
    # vis_im[index[0],index[1],:]=[0,0,0]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    # vis_im=cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR)

    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] +'_vpa.png', vis_parsing_anno)
        # cv2.imwrite(save_path[:-4] +'_pa.png', parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate_celeba(respth='./res/test_res', imgpth='./data', ckp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = get_model_5b(num_classes_rot=9,
                      num_classes_seg=19,
                      num_classes_cls=[2,7,4,6])
    net.cuda()

    net.load_param_multi_gpu(ckp)
    net.eval()

    to_tensor = transforms.Compose([
        # transforms.RandomResizedCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    num=0
    ch_id=[[0],[1],[2,3,4,5,6],[7,8,9],[10],[11,12,13],[14,15,16],[17,18]]
    with torch.no_grad():
        for image_path in os.listdir(imgpth):
            num+=1
            if num%2000==0:
                print("process NO.{} ----- {} ....".format(num, image_path))
            image = Image.open(osp.join(imgpth, image_path))
            
            # image = image.resize((224, 224), Image.BILINEAR)
            if image.size[0]!=224:
                raise ValueError('img size != 224')
            


            img = to_tensor(image)
            pr_img=torch.cat((img,img,img,img, img,img,img,img,img),0)
            pr_img=torch.unsqueeze(pr_img,0)
            pr_img=pr_img.cuda()
            img = torch.unsqueeze(img, 0)
            img = img.cuda()

            _,out,_,_,_,_ = net(img,pr_img)
            out = F.interpolate(input=out, size=(224, 224), mode='bilinear', align_corners=True)
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # 合并label
            # for newid,i in enumerate(ch_id):
            #     for j in i:
            #         # print(j)
            #         index = np.where(parsing == j)
            
            #         if index!=[]:
            #             # print(index)
            #             parsing[index[0],index[1]]=newid
            # print(parsing)
            # print(parsing.shape)
            # print(image_path)
            # print(np.unique(parsing))

            save_path=osp.join(respth, image_path)
            # print(parsing)
            # cv2.imwrite(save_path[:-4] +'.png', parsing)
            # if num%500==0:
            #     print(num)
            
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=save_path)
            # break
        print("total mask: ",num)

def evaluate_lfwa(imglst, respth='./res/test_res', imgpth='./data', ckp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = get_model_5b(num_classes_rot=9,
                      num_classes_seg=19,
                      num_classes_cls=[2,7,4,6])
    net.cuda()

    net.load_param_multi_gpu(ckp)
    net.eval()

    to_tensor = transforms.Compose([
        # transforms.RandomResizedCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    num=0
    ch_id = [[0], [1], [2, 3, 4, 5, 6], [7, 8, 9], [10], [11, 12, 13], [14, 15, 16], [17, 18]]
    
    with torch.no_grad():
        for image_path in imglst:
            num+=1
            if num%1000==0:
                print("process NO.{} ---- {} ...".format(num, image_path))
            image = Image.open(osp.join(imgpth, image_path))
            
            if image.size[0]!=224:
                raise ValueError('img size != 224')
            # image = img.resize((224, 224), Image.BILINEAR)

            img = to_tensor(image)
            pr_img=torch.cat((img,img,img,img,img,img,img,img,img),0)
            pr_img=torch.unsqueeze(pr_img,0)
            pr_img=pr_img.cuda()
            img = torch.unsqueeze(img, 0)
            img = img.cuda()

            _,out,_,_,_,_ = net(img,pr_img)
            out = F.interpolate(input=out, size=(224, 224), mode='bilinear', align_corners=True)
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            # 合并label
            # for newid,i in enumerate(ch_id):
            #     for j in i:
            #         # print(j)
            #         index = np.where(parsing == j)
            
            #         if index!=[]:
            #             # print(index)
            #             parsing[index[0],index[1]]=newid
            # print(parsing)
            # print(parsing.shape)
            # print(image_path)
            # print(np.unique(parsing))

            save_path=osp.join(respth, image_path.split('.')[0]+'.png')
            folder=osp.join(respth, image_path.split('/')[0])
            if not os.path.exists(folder):
                os.makedirs(folder)
            # crop.save(osp.join(respth, image_path))
            # print(parsing)
            # print(save_path)
            # cv2.imwrite(save_path, parsing) #存储生成的灰度seg图像
            # if num%100==0:
            #     print(num)
            
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=save_path)
            # break
        print("total mask: ", num)
        


def evaluate_maad(imglst, respth='./res/test_res', imgpth='./data', ckp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = get_model_5b(num_classes_rot=9,
                      num_classes_seg=19,
                      num_classes_cls=[2,7,4,6])
    net.cuda()

    net.load_param_multi_gpu(ckp)
    net.eval()

    to_tensor = transforms.Compose([
        # transforms.RandomResizedCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    num=0
    ch_id = [[0], [1], [2, 3, 4, 5, 6], [7, 8, 9], [10], [11, 12, 13], [14, 15, 16], [17, 18]]
    
    with torch.no_grad():
        for image_path in imglst:
            num+=1
            if num%2000==0:
                print("process NO.{} pic ---- {} ...".format(num, image_path))

            image = Image.open(osp.join(imgpth, image_path))
            
            if image.size[0]!=224:
                raise ValueError('img size != 224')
            # image = img.resize((224, 224), Image.BILINEAR)
            img = to_tensor(image)
            pr_img=torch.cat((img,img,img,img,img,img,img,img,img),0)
            pr_img=torch.unsqueeze(pr_img,0)
            pr_img=pr_img.cuda()
            img = torch.unsqueeze(img, 0)
            img = img.cuda()

            _,out,_,_,_,_ = net(img,pr_img)
            out = F.interpolate(input=out, size=(224, 224), mode='bilinear', align_corners=True)
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            # 合并label
            # for newid,i in enumerate(ch_id):
            #     for j in i:
            #         # print(j)
            #         index = np.where(parsing == j)
            
            #         if index!=[]:
            #             # print(index)
            #             parsing[index[0],index[1]]=newid
            # print(parsing)
            # print(parsing.shape)
            # print(image_path)
            # print(np.unique(parsing))

            save_path=osp.join(respth, image_path.split('.')[0]+'.png')
            folder=osp.join(respth, image_path.split('/')[0])
            if not os.path.exists(folder):
                os.makedirs(folder)
            # crop.save(osp.join(respth, image_path))
            # print(parsing)
            # print(save_path)
            # cv2.imwrite(save_path, parsing)
            # if num%200==0:
                # print(num)
            
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=save_path)
            # break
        print("total mask: ",num)



# -*- coding: utf-8 -*-

from PIL import Image

def get_test_img_attr(attr_file):
    with open(attr_file) as f:
        lines = f.readlines()
        set_size = len(lines)
        imgs = []
        # attr = np.zeros((set_size, 40))

        for i, line in enumerate(lines):
            vals = line.split()
            imgs.append(vals[0])
            # attr[i, :] = [vals[j + 1] for j in range(40)]
    return imgs


if __name__ == "__main__":
    
    # ckp='/media/data/sy_data/ckp/s3net_5b_newadv_newatt_pre_5_10/epoch17.pth'
    # ckp='/media/data/sy_data/ckp/s3net_5b_newadv_newatt_pre_5_10_maad/epoch11.pth'

    # ori_imgpth='/media/data/jrq_data/MAAD/train/train'
    # imgpth='/media/data/sy_data/DataSet/MAAD/crop75_train_20w'
    # savepth = '/media/data/sy_data/DataSet/MAAD/crop75_train_20w_mask_19'
    # attrpth = '/media/data/sy_data/DataSet/MAAD/anno/train_attr_file_20w.npy'  #6263pics
    
    # ori_imgpth='/media/data/jrq_data/MAAD/test'
    # imgpth='/media/data/sy_data/DataSet/MAAD/crop75_test_2w'
    # savepth = '/media/data/sy_data/DataSet/MAAD/crop75_test_2w_mask_19'
    # attrpth='/media/data/sy_data/DataSet/MAAD/anno/test_attr_file_2w.npy' #6880pics


    # ==================== for maad ==================================
    savepth = '/home/sy/segment_pic/maad_no_smooth_19'
    ori_imgpth='/media/data/jrq_data/MAAD/train/train'
    imgpth='/media/data/sy_data/DataSet/MAAD/crop75_train_20w'
    attrpth = '/media/data/sy_data/DataSet/MAAD/anno/train_attr_file_20w.npy'  #6263pics

    # if not os.path.exists(imgpth):
    #     os.makedirs(imgpth)

    num=0
    train_attr_file = np.load(attrpth, allow_pickle=True)
    image_names = train_attr_file[:, 0]
   
    # import shutil
    # for image_path in image_names:
            
    #     num+=1
    #     file_path = osp.join(ori_imgpth, image_path)
    #     img = Image.open(file_path)

    #     crop=transforms.Resize(224)(transforms.RandomCrop(50)(transforms.Resize(100)(img)))
    #     folder=osp.join(imgpth, image_path.split('/')[0])
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
            
    #     new_file_path=osp.join(imgpth, image_path)
    #     crop.save(new_file_path)
    #     if num%500==0:
    #         print(num)
    #         # break
    #     # if num == 2000:
    #     #     break
    # #-----------------------------------------------
    # print("total pics: ",num)    
    print("img_pth is {}, \nsave_pth is {}".format(imgpth, savepth))
    evaluate_maad(image_names,savepth,imgpth,ckp='/home/sy/segment_pic/ckp/maad_epoch11.pth')

    #======================================= for lfwa and celeba ===========================


    

    # # ----------------------------- for celeba
    # dspth='/media/sy/disk/DataSet/CelebA/img_align_celeba'
    # imgpth='/media/data/sy_data/DataSet/CelebA/img_align_celeba_crop75'
    # savepth='/home/sy/segment_pic/celeba_no_smooth_19'
    # respth='' # 生成新的图像的存储地址
    # num=0

    # # for image_path in os.listdir(respth): 
    # #         num+=1
    # #         img = Image.open(osp.join(respth, image_path))
    # #         # crop=transforms.Resize(224)(transforms.RandomCrop(75)(img))
    # #         crop=transforms.Resize((224,224),Image.BILINEAR)(img)
    # #         crop.save(osp.join(respth, image_path))
    # #         # if num%100==0:
    # #             # print(num)
    # #             # break
    # print("img_pth is {}, \nsave_pth is {}".format(imgpth, savepth))
    # evaluate_celeba(savepth, imgpth, ckp='/home/sy/segment_pic/ckp/celeba_lfwa_epoch17.pth')
    # # -----------------------------------------------


    # # -----------------------------------for lfwa
    # dspth='/media/sy/disk/DataSet/LFW/lfw-deepfunneled'
    # respth='' # 生成新的图像的存储地址
    # imgpth='/media/data/sy_data/DataSet/LFW/lfw_crop75'
    # savepth='/home/sy/segment_pic/lfw_no_smooth_19' # 生成的分割图像的存储地址
    # trainpth='/media/data/sy_data/DataSet/LFW/anno/train.txt' #6263pics
    # testpth='/media/data/sy_data/DataSet/LFW/anno/test.txt' #6880pics

    # if not os.path.exists(savepth): #存储结果的文件夹
    #     os.makedirs(savepth)

    # imglst=get_test_img_attr(trainpth) 

    # # #----------------------- 根据原始图像生成新的裁剪之后的图像并存储 开始 ------------------------
    # # import shutil
    # # num=0
    # # for image_path in imgpth:
            
    # #     num+=1
    # #     # img = Image.open(osp.join(respth, image_path))
    # #     file_path=osp.join(respth, image_path)
    # #     # crop=transforms.Resize(224)(transforms.RandomCrop(120)(img))
    # #     # folder=osp.join(respth, image_path.split('/')[0])
    # #     # if not os.path.exists(folder):
    # #     #     os.makedirs(folder)
    # #     name=str(num)+'.jpg'
    # #     new_file_path=osp.join(savepth, name)
    # #     shutil.copy(file_path,new_file_path) 
    # #     # crop.save(osp.join(respth, image_path))
    # #     if num==500:
    # #         print(num)
    # #         break

    # # print("total pics: ",num)    
    # # #----------------------- 根据原始图像生成新的裁剪之后的图像并存储 结束 ------------------------

    
    # evaluate_lfwa(imglst, savepth, imgpth, ckp='/home/sy/segment_pic/ckp/celeba_lfwa_epoch17.pth')
    # # ---------------------------- lfwa end -----------------------------------------


        

