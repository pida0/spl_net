'''
Author: sy
Date: 2021-03-30 09:29:18
LastEditors: your name
LastEditTime: 2022-06-11 19:03:56
Description: file content
'''
# #=========== test cat ===============
# import torch
# x = torch.randn((2, 3, 5, 5))
# y = torch.cat((x, x, x), 1)
# print(x.shape, y.shape)

#============== test clone =======
# import torch
# import numpy as np
# acc = np.array([0, 0, 0, 0], dtype='float64')
# print(acc.dtype)
# a = torch.randn((2, 3, 5, 5))
# b = torch.randn((2, 3, 5, 5))
# c = a * b
# print(c,c.shape)
# correct = torch.FloatTensor(3, 2).fill_(0)
# print(correct[1][1])
# print ( '{name},{sex},{age}' . format (age = 32 ,sex = 'male' ,name = 'zhangk' ))
# a = torch.randn((3, 5, 5))
# b = [a for i in range(9)]
# prt_data_l = [0] * 3
# for i in range(3):
#     prt_data_l[i] = torch.cat(b[(i * 3):(i * 3 + 3)], 0)
# prt_data_l = torch.stack((prt_data_l))
# c = torch.stack((prt_data_l, prt_data_l, prt_data_l, prt_data_l))
# c=c.permute(1, 0, 2, 3, 4)
# print(prt_data_l.shape,c.shape)

#====================== test dataset ===================
# from configs import cfg
# from datasets import CelebA
# cfg_pth='/home/sy/s3net/config_files/test.yml'
# cfg.merge_from_file(cfg_pth)

#     # data_path = '/media/data/sy_data/DataSet/CelebA/Eval/list_eval_partition.txt'
#     # attr_path = '/media/data/sy_data/DataSet/CelebA/Anno/list_attr_celeba.txt'
#     # img_path = '/media/data/sy_data/DataSet/CelebA/img_align_celeba_crop75/'
#     # mask_path = '/media/data/sy_data/DataSet/CelebA/crop75_mask/'
#     # image_size = (227, 227)


# data_set = CelebA(cfg.DATA, '0', 'pretext-train')
#     # data_set = CelebA(data_path, attr_path, '0', img_path, mask_path, 'pretext-train', 5, 1)
# testloader = torch.utils.data.DataLoader(data_set, batch_size=10, shuffle=False)

# # for i in range(10):
# #     data,label=data_set[i]
# #     print(len(data))
# #     print(len(label))
#         # print(attr.shape)
#         # print(label)
# data, label = data_set[0]
# for i in range(len(data)):
#     print(data[i].shape)
#     print(label[i].shape)

# for b, (data, label) in enumerate(testloader):
#     if (b == 1):
#         break
#     print(data[0].shape,data[1].shape,data[2].shape)
#     print(label[0].shape,label[1].shape,label[2].shape)


#================= test group ========================
# import numpy as np
# group_lst = {13: [1, 3, 4, 5, 8, 9, 11, 12, 15, 17, 23, 28, 35],
#              6: [7, 19, 27, 29, 30, 34],
#              9: [6, 14, 16, 21, 22, 24, 36, 37, 38],
#              12: [0, 2, 10, 13, 18, 20, 25, 26, 31, 32, 33, 39]}
# group = [12, 13, 6, 9]
# attr = np.random.randn(3, 40)
# print(attr)

# attrs = [0] * 4
# for i in range(len(group)):
#     attrs[i] = attr[0, group_lst[group[i]]]

# print(len(attrs))

#================== test dis =======================
# import torch
# import torch.nn as nn
# from models import Discriminators

# D=Discriminators()

# x = torch.randn(16, 512)
# out = D(x)
# label = torch.zeros(out.size())
# cri = nn.BCEWithLogitsLoss()

# loss = cri(out, label)
# print(loss)

#=================== test size 75 =================
import torch
import torch.nn as nn
from models import get_model_5b

model = get_model_5b(9,19)

    # print(model)
x = torch.randn(24, 3, 75, 75)
y = torch.randn(24, 27, 224, 224)
# z = torch.randn(16, 9, 224, 224)
# z = torch.stack((z, z, z))
    
x = x.cuda()
y = y.cuda()
# z = z.cuda()
# print(z.size())
model=model.cuda()
prt, pst, pct, x_prtg_fake, x_prtg_real= model(x,y)
print(prt.shape,pst.shape,pct.shape, x_prtg_fake.shape,x_prtg_real.shape)