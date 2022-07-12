'''
Author: sy
Date: 2021-10-13 15:44:08
LastEditors: your name
LastEditTime: 2021-11-06 21:09:23
Description: file content
'''

# from models.seg_self_model import SSM
from matplotlib.pyplot import yticks
from torch import randn
from torchvision.transforms.transforms import RandomCrop, Resize
# from datasets.utils import random_crop
import torch
from torchvision import models
# fcn=models.segmentation.fcn_resnet101(True).eval()

from PIL import Image
import matplotlib.pyplot as plt
# img=Image.open('/media/sy/disk/DataSet/CelebA/000001.jpg')
# # img=img.convert('RGB')
# print(img.size[0])

# # # plt.imshow(img);plt.show()

# import torchvision.transforms as T
# trf=T.Compose([
#     # T.Resize(512),
#     # T.RandomCrop(75),
#     T.Resize(75),
#     # T.ToTensor(),
#     # T.Normalize([0.485, 0.456, 0.406],
#     #             [0.229, 0.224, 0.225])
# ])

# inp=trf(img)
# print(inp.size[0])
# plt.imshow(inp);plt.show()
# print("input size: ",inp.shape)

# out=fcn(inp)
# print("out type: ",type(out))
# print("out keys: ",out.keys())
# print("out size: ",out['out'].shape) # torch.Size([1, 21, 224, 224])
# print("aux size: ",out['aux'].shape) # torch.Size([1, 21, 224, 224])

# import numpy as np
# om=torch.argmax(out['out'].squeeze(),dim=0).detach().cpu().numpy()
# print("om size: ",om.shape)
# print("class: ",np.unique(om))

# # Define the helper function
# def decode_segmap(image, nc=21):
   
#   label_colors = np.array([(0, 0, 0),  # 0=background
#                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
#                (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
#                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
#                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
#                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
#                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
#                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
#                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
 
#   r = np.zeros_like(image).astype(np.uint8)
#   g = np.zeros_like(image).astype(np.uint8)
#   b = np.zeros_like(image).astype(np.uint8)
   
#   for l in range(0, nc):
#     idx = image == l
#     r[idx] = label_colors[l, 0]
#     g[idx] = label_colors[l, 1]
#     b[idx] = label_colors[l, 2]
     
#   rgb = np.stack([r, g, b], axis=2)
#   return rgb

# rgb=decode_segmap(om)
# plt.imshow(rgb);plt.show()

#============================================
# a=torch.randn((3,2,2))
# # a=torch.tensor([[[-0.9268, 0.6006],
# # [ 1.0213, 0.5328]],
# # [[-0.7024, 0.7978],
# # [ 1.0553, -0.6524]]])
# print("=========PIL=====")
# ap=T.ToPILImage(a)
# print(ap)
# # print(ap.shape)
# print("==========trans tensor=============")
# at=a.transpose(1,2)
# print(at)
# print(at.shape)
# print("============trans PIL==============")
# apt=T.RandomVerticalFlip(p=1)(ap)
# print(apt)
# img=T.CenterCrop(3)(img)

# # print(T.ToTensor()(img))
# print(T.ToTensor()(img).transpose(1,2))
# t=T.RandomVerticalFlip(p=1)(img)
# print(T.ToTensor()(t))
# from models.seg_self_model import SSM
# net=SSM(8,2,model_name='resnet50')
# # print(net)
# input=torch.rand((1,3,512,512))
# out=net(input)
# print(out[0].shape,out[1].shape)


#===================test celeba.py==================
# from datasets.celeba import CelebA
# data_path = '/media/sy/disk/DataSet/CelebA/Eval/list_eval_partition.txt'
# attr_path = '/media/sy/disk/DataSet/CelebA/Anno/list_attr_celeba.txt'
# img_path='/media/sy/disk/DataSet/CelebA/img_align_celeba_crop75/'
# mask_path='/media/sy/disk/DataSet/CelebA/crop75_mask/'
# image_size = (227, 227)
# tool_pth='/media/sy/disk/Code/face-parsing.PyTorch-master/res/cp/79999_iter.pth'
# mode = 'train'

# data_set = CelebA(data_path, attr_path, '0', img_path, mask_path, 'self-train-rot', 5, 1)
# testloader = torch.utils.data.DataLoader(data_set, batch_size=10, shuffle=False)

# for i in range(10):
#     data,label,attr,rot=data_set[i]
#     print(data.shape)
#     print(label.shape,type(label))
#     print(attr.shape)
#     print(rot)
#     print('===========================')
    
# cls_label_count=torch.zeros((8))
# for batch_idx, (images, labels, attrs) in enumerate(testloader):
#   # print(batch_idx)
#   # print(images.shape)
#   # # print(labels.shape)
#   # print(attrs.shape)
#   if batch_idx%1000==0:
#     print(batch_idx)
#   cls_label_count+=attrs.sum(0)
  
# print("label count: ", cls_label_count)  
# ds_len=len(data_set)
# ratio=cls_label_count/(ds_len)
# print("ratio: ", ratio)


#=================test count=====================
# import numpy as np

# a=np.array([[2,1,2],[3,2,3],[4,1,4]])
# ll=[]
# for i in range(6): #0-1,2-3-4,5
#   ll.append(sum(sum(a==i)))
# print(ll)

# ch_id=[[0,1],[2,3,4],[5]]

# cls=[]
# for id in (ch_id):
#   print(id)
#   sum_=0
#   for i in id:
#     sum_+=ll[i]
#   cls.append(sum_)
# print(cls)
# _,pre=torch.tensor(cls).topk(2,dim=0,largest=True)
# print(pre)

# cls_label=torch.zeros((1,8))
# for pos in pre:
#   cls_label[0][pos]=1

# print(cls_label)

#==================== test crossentropy =================
# import torch.nn as nn
# import torch
# inp=torch.randn((1,4,5,5))
# target=torch.randint(0,4,(1,5,5))
# print("inp: ",inp)
# print("target: ",target)

# cri=nn.CrossEntropyLoss()
# loss=cri(inp,target)
# print("loss: ",loss)

#============================= test raise =============
# a=5
# if a==4:
#   print(a)
# else:
#   raise ValueError('a != 4')
# print(a)


#============================ test seg_self_model ============
# from models import SSM
# from models.resnet import ResNet
# import torch
# net=ResNet(50)
# # print(len(net.state_dict().keys()))
# load_epoch=0
# model_path='/media/sy/disk/ckp/seg_ssl_v1/epoch{}.pth'.format(load_epoch)
# premodel=torch.load(model_path)
# # print(premodel.keys())
# num=0
# for i in net.state_dict():
#   if 'classifier' in i:
#     # print(pre_key)
#     # print(i)
#     # print(net.state_dict()[i])
#     # print(premodel[i])
#     # print(num)
#     # print('===============================')
#     # net.state_dict()[i].copy_(premodel[i])
#     continue
#   num+=1
#   print(i)
#   # pre_key='base.'+i
#   # print(pre_key)
#   # print(net.state_dict()[i])
#   # print(premodel[pre_key])
#   # print(num)
#   # print('===============================')
#   net.state_dict()[i].copy_(premodel['base.'+i])
# print(num)
# # for idx,(name, module) in enumerate(net._modules.items()):
# #     print(idx,'--',name,' ========== ',module)

#===================== test seg-self model =============
# from models import SSM1,SSM2
# import torch
# from PIL import Image
# from datasets.celeba import CelebA
# data_path = '/media/sy/disk/DataSet/CelebA/Eval/list_eval_partition.txt'
# attr_path = '/media/sy/disk/DataSet/CelebA/Anno/list_attr_celeba.txt'
# img_path='/media/sy/disk/DataSet/CelebA/img_align_celeba_crop75/'
# mask_path='/media/sy/disk/DataSet/CelebA/crop75_mask/'
# image_size = (227, 227)
# tool_pth='/media/sy/disk/Code/face-parsing.PyTorch-master/res/cp/79999_iter.pth'
# mode = 'train'

# data_set = CelebA(data_path, attr_path, '0', img_path, mask_path, 'self-train', 5, 1)
# testloader = torch.utils.data.DataLoader(data_set, batch_size=10, shuffle=False)

# for i in range(10):
#     data,label,attr=data_set[i]
#     print(data.shape)
#     print(label.shape,type(label))
#     print(attr.shape)
#     print('===========================')

# img=Image.open('bird.jpg')
# # img=img.convert('RGB')
# # print(img.size[0])

# # # plt.imshow(img);plt.show()
# import datetime
# import torchvision.transforms as T
# trf=T.Compose([
#     # T.Resize(75),
#     T.RandomCrop(224),
#     # T.Resize(512),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406],
#                 [0.229, 0.224, 0.225])
# ])
# inp=trf(img)
# # print(inp.shape)
# inp=inp.unsqueeze(0)

# net=SSM2(8)
# print(net)
# print(inp.shape)

# start_time=datetime.datetime.now()
# x_self,x_seg=net(inp)
# end_time=datetime.datetime.now()
# print('run time %.2f s' % ((end_time-start_time).total_seconds()))
# print(x_self.shape)
# print(x_seg.shape)

#===================== test lambda lr ==================
# from utils import lambda_warm_up
# import torch.optim as optim
# import torch.nn as nn
# from torch.optim import lr_scheduler

# class net(nn.Module):
#     def __init__(self):
#         super(net, self).__init__()
#         self.fc1=nn.Linear(32,12)
#         self.relu=nn.ReLU()
#         self.fc2=nn.Linear(12,2)
#     def forward(self,x):
#         x=self.fc2(self.relu(self.fc1(x)))
#         return x

# model=net().cuda()
# print(model)
# cri= nn.CrossEntropyLoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, betas=(0.9, 0.999))
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warm_up)

# for epoch in range(60):
#     print('epoch {}: lr = {}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
#     inp=torch.rand((20,32)).cuda()
#     label=torch.randint(0,2,(20,1)).cuda().squeeze(1)
#     out=model(inp)
#     # print(out.shape,label.shape)
#     loss=cri(out,label)
#     loss.backward()
#     optimizer.step()
#     scheduler.step()

#================== test write cfg ================
# from configs import cfg, show_cfg
# import argparse

# cfg.merge_from_file('./config_files/train_attr.yml')

# cfg.freeze()
# from tensorboardX import SummaryWriter as sw
# writer = sw(cfg.SW_LOG_DIR)
# writer.add_text('cfg',str(cfg))      
# writer.close()

#================= test random ============
# import numpy as np
# rot_label = np.random.randint(0,4,(20000,1))
# # id = np.random.randint(0,19999,1)
# for i in range(10):
#     id = int(np.random.randint(0,19999,1))
#     print(rot_label[id])

#================= test lfwa ==================
# from datasets.lfwa import LFWA
# from configs import cfg, show_cfg
# from datasets import make_dataloader

# data_path = '/media/sy/disk/DataSet/CelebA/Eval/list_eval_partition.txt'
# attr_path = '/media/sy/disk/DataSet/CelebA/Anno/list_attr_celeba.txt'
# img_path='/media/sy/disk/DataSet/CelebA/img_align_celeba_crop75/'
# mask_path='/media/sy/disk/DataSet/CelebA/crop75_mask/'
# image_size = (227, 227)
# config_file='./config_files/train_attr.yml'
# mode = 'train'

# cfg.merge_from_file(config_file)
# trainloader, valloader, testloader = make_dataloader(cfg)
# # data_set = CelebA(data_path, attr_path, '0', img_path, mask_path, 'self-train-rot', 5, 1)
# # testloader = torch.utils.data.DataLoader(data_set, batch_size=10, shuffle=False)

# for batch_idx, (images, seg) in enumerate(trainloader):
    
#     print(images.shape)
#     print(seg.shape)
#     # print(cls.shape)
#     # print(rot.shape)
#     # print(attr.shape)
#     # print(rot)
#     # if batch_idx ==10:
#     #     break
#     print('============ {} ==============='.format(batch_idx))
    
# cls_label_count=torch.zeros((8))
# ds_len=0
# for batch_idx, (images, labels, attrs, rot) in enumerate(trainloader):
#   # print(batch_idx)
#   # print(images.shape)
#   # # print(labels.shape)
#   # print(attrs.shape)
#   if batch_idx%50==0:
#     print(batch_idx)
#   ds_len+=rot.size()[0]
#   cls_label_count+=attrs.sum(0)
  
# print("label count: ", cls_label_count)  
# # ds_len=len(data_set)
# print('ds_len: ',ds_len)
# ratio=cls_label_count/(ds_len)
# print("ratio: ", ratio)

#==============test module============
# from configs import cfg, show_cfg
# cfg_pth_s_pr = './config_files/seg_ssl_pr.yml'
# cfg.merge_from_file(cfg_pth_s_pr)
# from datasets import make_dataloader
# trainloader, valloader, testloader = make_dataloader(cfg)

# for batch_idx, (pr_imgs, images, seg_labels, cls_labels, pr_labels) in enumerate(trainloader):
#   # print(batch_idx)
#   print(images.shape, pr_imgs.shape, seg_labels.shape, cls_labels.shape, pr_labels.shape)

#   # # print(labels.shape)
#   # print(attrs.shape)
#   if batch_idx%2==0:
#     # print(batch_idx)
#     break
# #   ds_len+=rot.size()[0]
# #   cls_label_count+=attrs.sum(0)


#============================== plot correlation ================================
import torch
import torch.nn as nn
import torch.optim as optim
from models import get_down_model, get_down_model_5b, PreActResNet18
import os
import argparse
from torch.optim import lr_scheduler
import datetime
from tensorboardX import SummaryWriter as sw
from utils import cal_attr_acc, lambda_warm_up, BalancedLoss
from configs import cfg, show_cfg
from datasets import make_dataloader,CelebA, LFWA,MAAD
import numpy as np
import seaborn as sns

group_lst_down = [0, 2, 10, 13, 18, 20, 25, 26, 31, 32, 33, 39, 1, 3, 4, 5, 8, 9, 11, 12, 15, 17, 23, 28, 35, 7, 19, 27, 29, 30, 34, 6, 14, 16, 21, 22, 24, 36, 37, 38]

name=['5_o_clock_shadow',
'Arched_Eyebrows',
'attractive',
'bags_under_eyes',
'bald',
'bangs',
'big_lips',
'big_nose',
'black_hair',
'blond_hair',
'blurry',
'brown_hair',
'bushy_eyebrows',
'chubby',
'double_chin',
'eyeglasses',
'goatee',
'gray_hair',
'heavy_makeup',
'high_cheekbones',
'male',
'mouth_slightly_open',
'mustache',
'narrow_eyes',
'no_beard',
'oval_face',
'pale_skin',
'pointy_nose',
'receding_hairline',
'rose_cheeks',
'sideburns',
'smiling',
'straight_hair',
'wavy_hair',
'earrings',
'hat',
'lipstick',
'necklace',
'necktie',
'young']

name=[name[group_lst_down[i]]for i in range(0,40)]


# parser = argparse.ArgumentParser(description="Seg_SSL_v3 Training")
# parser.add_argument(
#         "--config_file", default='./config_files/train_downstream.yml', help="path to config file", type=str
# )
# parser.add_argument("opts", help="Modify config options using the command-line", default=None,
#                         nargs=argparse.REMAINDER)

# args = parser.parse_args()

# cfg.merge_from_file(args.config_file)
# cfg.freeze()


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# group1=[12, 13, 6, 9]
# group2=[17,16,6,8] #MAAD

# # trainset = CelebA(cfg.DATA.DATASET_NAME, cfg=cfg.DATA, partition='0', type=cfg.DATA.TYPE + '-train', group=group1)
# testset = CelebA(cfg=cfg.DATA, partition='2', type=cfg.DATA.TYPE + '-val', group=group1)

# # trainset = LFWA(cfg.DATA.DATASET_NAME, cfg=cfg.DATA, partition='0', type=cfg.DATA.TYPE + '-train', group=group1)
# # testset = LFWA(cfg.DATA.DATASET_NAME, cfg=cfg.DATA, partition='2', type=cfg.DATA.TYPE + '-val', group=group1)

# # trainset = MAAD(cfg.DATA.DATASET_NAME, cfg=cfg.DATA, partition='0', type=cfg.DATA.TYPE + '-train', group=group2)
# # testset = MAAD(cfg.DATA.DATASET_NAME, cfg=cfg.DATA, partition='2', type=cfg.DATA.TYPE + '-val', group=group2)

# model = PreActResNet18(40)
# # model = get_down_model_5b(num_classes_cls=[12, 13, 6, 9])
# # model = get_down_model_5b(num_classes_cls=[17,16,6,8],dataset='MAAD')

# gt_label = [] #np.zeros((len(trainset), 40))
# pre_label=[]

  


# model_path=cfg.TRAIN.LOAD_PTH # + 'epoch{}.pth'.format(cfg.TRAIN.LOAD_EPOCH)
# # model_path=cfg.TRAIN.SAVE_PTH + 'epoch{}.pth'.format(cfg.TRAIN.PRETRAIN_EPOCH)
# model.load_state_dict(torch.load(model_path))
# # model.load_param_multi_gpu(model_path)

# # model_path=cfg.TRAIN.SAVE_PTH + 'epoch{}.pth'.format(cfg.TRAIN.PRETRAIN_EPOCH)
# # model.load_param_from_segself(model_path)
        
# print('Model loaded from {}......'.format(model_path))

# model.cuda()
# model.eval()
        
        
# with torch.no_grad():
            
#     for idx, (images, attrs) in enumerate(testset):
        
#         if idx%1000==0:
#           print(idx)
#         images = images.cuda()
#         images.unsqueeze_(0)
#         # print(images.shape)
#         output = model(images)
#         # total_attr=np.append(attrs[0],attrs[1])
#         # total_attr=np.append(total_attr,attrs[2])
#         # total_attr=np.append(total_attr,attrs[3])
#         # gt_label.append(total_attr.tolist())
#         # pre_re.append(output.squeeze_().cpu().numpy().tolist())
#         # output要做合并处理，并且要注意预测出来的结果的顺序
#         # total_out=torch.cat(output,dim=1)
#         com1=output.sigmoid()
#         pre_label.append(com1.squeeze_().cpu().numpy().tolist())

# print('gt_label',len(gt_label))
# # print(gt_label)        
# print('pre_label',len(pre_label))
# # print(pre_label) 
# # print('pre_re',len(pre_re))
# # print(pre_re)    

# # gt_label_mat=np.array(gt_label)
# # pre_re_mat=np.array(pre_re)
# pre_label_mat=np.array(pre_label)

# # np.savetxt("/home/sy/s3net/relation_files/celeba_gt_label.txt",gt_label_mat)

# np.savetxt("/home/sy/s3net/relation_files/celeba_test_pre_label_baseline.txt",pre_label_mat)

#============================= 画图 ======================================

# gt_label_mat=np.loadtxt('./heatmap_files/celeba_gt_label.txt')
pre_label_mat=np.loadtxt('./relation_files/celeba_sspl_label.txt')
baseline_pre_label_mat=np.loadtxt('./relation_files/celeba_baseline_label.txt')

print_idx= [25,3,29,17,18,39,11]#[11,35,8,9,4,25,3] # [0,4,18,28,35,36,39]
# pre_label_mat = pre_label_mat[print_idx]    #先取出想要的行数据
pre_label_mat = pre_label_mat[:,print_idx] #再取出要求的列数据
# baseline_pre_label_mat = baseline_pre_label_mat[print_idx]    #先取出想要的行数据
baseline_pre_label_mat = baseline_pre_label_mat[:,print_idx] #再取出要求的列数据
name=np.array(name)[print_idx]
print(name)



# print(gt_label_mat.shape)
# gt_label_relation=np.corrcoef(gt_label_mat,rowvar=False)
# pre_re_relation=np.corrcoef(pre_re_mat,rowvar=False)
pre_label_relation=np.corrcoef(pre_label_mat,rowvar=False)
baseline_pre_label_relation=np.corrcoef(baseline_pre_label_mat,rowvar=False)


# offset=baseline_pre_label_relation-pre_label_relation
# abs_offset=abs(offset)
# idx=np.argwhere(abs_offset>0.55)
# uni_idx=np.unique(idx.ravel())
# print(uni_idx.shape, uni_idx)
# print(np.array(name)[uni_idx])

# ax=plt.subplot(1,1,1)
# sns.heatmap(abs_offset,square=True,annot=False,ax=ax,xticklabels=name, yticklabels=name, cmap='PuBu')
# plt.show()


# plt.clf()
plt.figure(1, figsize=(15, 15))
ax1=plt.subplot(1,2,1)
ax1.set_title('(a) baseline',y=-0.69)
ax1.tick_params(labelsize=18)

sns.heatmap(baseline_pre_label_relation,square=True,annot=True,xticklabels=name, yticklabels=name, 
            cmap='PuBu',cbar_kws={"shrink": .4},annot_kws={'size':10})

cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
plt.title('(a) baseline',fontdict={'weight':'normal','size': 30})
# plt.xticks(rotation=20) 
# plt.show()

ax2=plt.subplot(1,2,2)
ax2.set_title('(a) SPL-Net',y=-0.69)
ax2.tick_params(labelsize=18)
sns.heatmap(pre_label_relation,square=True,annot=True,ax=ax2, xticklabels=name, yticklabels=name, 
            cmap='PuBu',cbar_kws={"shrink": .4},annot_kws={'size':10})
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
plt.title('(b) SPL-Net',fontdict={'weight':'normal','size': 30})

plt.subplots_adjust(top=1.0,
bottom=0.28,
left=0.155,
right=0.955,
hspace=0.265,
wspace=0.5)
# top=1.0,
# bottom=0.165,
# left=0.11,
# right=0.9,
# hspace=0.2,
# wspace=0.345)

plt.savefig('./relation_files/relation.pdf',format='pdf')
plt.show()


 
             
        
  
        
        
       