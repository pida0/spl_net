'''
Author: sy
Date: 2021-04-01 09:41:43
LastEditors: your name
LastEditTime: 2022-06-25 07:23:55
Description: file content
'''
import numpy as np
from sqlalchemy import false, true
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms.transforms import RandomCrop
import torch
from PIL import Image

#=============== for celeba ====================== 
def make_img(part_dir, partition):
    imglist = []
    with open(part_dir) as f:
        lines = f.readlines()
        for line in lines:
            pic_dir, num = line.split()
            if num == partition: 
                imglist.append(pic_dir)
    return imglist

def get_attr(pth):
    attrs = np.zeros((202600, 40))
    with open(pth) as f:
        f.readline()
        f.readline()
        lines = f.readlines()
        id = 0
        for line in lines:
            vals = line.split()
            id += 1
            for j in range(40): #(-1,1)-->(0,1)
                attrs[id, j] = (int(vals[j + 1]) + 1) / 2
    return attrs

def rgb_jittering(im):  #对每个通道的像素值进行变换
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')

def get_prt_data_label(prt_img, full_mask, patch_num=9):
    prt_image_trans = T.Compose([T.Resize(512, Image.BILINEAR)])
    if prt_img.size[0] != 512 or full_mask.size[0]!=512:
        raise ValueError('img size is {}, mask size is {}. size need to be 512.'.format(prt_img.size[0], full_mask.size[0]))

    crop_size = 150
    if patch_num==4:
        crop_size = 236
    elif patch_num==16:
        crop_size=100
    
    prt_tile_trans = T.Compose([T.Resize((224, 224), Image.BILINEAR),
                                T.Lambda(rgb_jittering),
                                T.ToTensor()])
                                
    
    img_tiles = [None] * patch_num
    is_full_bg = [0] * patch_num
    s = float(prt_img.size[0]) / int(patch_num ** 0.5)  # 3 # 2
    a = s / 2
    
    for n in range(patch_num): # 4
        i = n / int(patch_num ** 0.5)  # 2
        j = n % int(patch_num ** 0.5)  # 2
        c = [a * i * 2 + a, a * j * 2 + a]
        c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
        img_tile = prt_img.crop(c.tolist())
        mask_tile = full_mask.crop(c.tolist())
        
        # if n == rot_id:
        #     tile = T.RandomVerticalFlip(p=1)(tile)
        seed=torch.random.seed()
        torch.random.manual_seed(seed)
        img_tile = T.RandomCrop(crop_size)(img_tile)
        torch.random.manual_seed(seed)
        mask_tile = T.RandomCrop(crop_size)(mask_tile)
        mask_tile = np.array(mask_tile).astype(np.int64)
        is_full_bg[n]=(np.sum(mask_tile)<crop_size*crop_size/4)

        img_tile = prt_tile_trans(img_tile)

        # Normalize the patches indipendently to avoid low level features shortcut
        m, s = img_tile.view(3, -1).mean(dim=1).numpy(), img_tile.view(3, -1).std(dim=1).numpy()
        s[s == 0] = 1
        norm = T.Normalize(mean=m.tolist(), std=s.tolist())
        img_tile = norm(img_tile)
        # print(tile.shape)
        img_tiles[n] = img_tile

    rot_id=0
    rot_angle=0
    cnt=0
    while(true):
        rot_id=np.random.randint(0, patch_num)
        cnt+=1
        if not is_full_bg[rot_id] or cnt>=patch_num:
            rot_angle=np.random.randint(0,4)
            img_tiles[rot_id]=random_rotate(img_tiles[rot_id], rot_angle)
            break
    prt_data_g = torch.cat(img_tiles, 0)
    # prt_data_l = [0] * 3
    # for i in range(3):
    #     prt_data_l[i] = torch.cat(tiles[(i * 3):(i * 3 + 3)], 0)
    # prt_data_l = torch.stack(prt_data_l)
      
    return prt_data_g, rot_id, rot_angle

def get_prt_data(prt_img, prt_id, patch_num=9):
    prt_image_trans = T.Compose([T.Resize(256, Image.BILINEAR),
                                 T.CenterCrop(255)])
    if prt_img.size[0] != 255:
        prt_img = prt_image_trans(prt_img)

    crop_size = 64
    if patch_num==4:
        crop_size = 100
    elif patch_num==16:
        crop_size=48
    
    prt_tile_trans = T.Compose([T.RandomCrop(crop_size), #86->64, 128->100
                                T.Resize((224, 224), Image.BILINEAR),
                                T.Lambda(rgb_jittering),
                                T.ToTensor()])
    
    tiles = [None] * patch_num
    s = float(prt_img.size[0]) / int(patch_num ** 0.5)  # 3 # 2
    a = s / 2
    
    for n in range(patch_num): # 4
        i = n / int(patch_num ** 0.5)  # 2
        j = n % int(patch_num ** 0.5)  # 2
        c = [a * i * 2 + a, a * j * 2 + a]
        c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)
        tile = prt_img.crop(c.tolist())
        if n == prt_id:
            tile = T.RandomVerticalFlip(p=1)(tile)
        tile = prt_tile_trans(tile)

        # Normalize the patches indipendently to avoid low level features shortcut
        m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
        s[s == 0] = 1
        norm = T.Normalize(mean=m.tolist(), std=s.tolist())
        tile = norm(tile)
        # print(tile.shape)
        tiles[n] = tile

    prt_data_g = torch.cat(tiles, 0)
    # prt_data_l = [0] * 3
    # for i in range(3):
    #     prt_data_l[i] = torch.cat(tiles[(i * 3):(i * 3 + 3)], 0)
    # prt_data_l = torch.stack(prt_data_l)
      
    return prt_data_g


def pil_loader(path, type):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if type=='img':
                return img.convert('RGB')
            elif type=='mask':
                return img.convert('P')
            else:
                raise ValueError('type {} not supported. Only support img、mask'.format(type))


def build_transform(type='self',i=0,j=0,th=75,tw=75):
    '''
    Descripttion: definition for transform
    Param: 
    Return: 
    '''
    if type=='pretext-train' or type=='pretext-train' :
        # 图像已经crop过了
        trans=T.Compose([# (224,224)
            # T.RandomCrop(75),
            # random_crop(i, j, th, tw),
            # T.Resize(512),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif type=='pretext-val' or type=='self-rot-val' :
        trans=T.Compose([
            # T.RandomCrop(75),
            # random_crop(i, j, th, tw),
            # T.Resize(512),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif type=='self-pr-val' or type=='self-pr-train':
        trans=T.Compose([
            # T.Resize(75),
            # random_crop(i, j, th, tw),
            # T.Resize(512),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif type=='downstream-train':
        trans=T.Compose([
            # 下游任务的图像处理，就是属性识别的图像处理
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomGrayscale(),
            T.RandomRotation((30, 45)),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif type=='downstream-val':
        trans=T.Compose([
            T.Resize((224, 224)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        raise ValueError('type {} not supported. Only support self-train、self-val、fine-tune-train、fine-tune-val'.format(type))

    return trans


def get_labels(seg_label, topk_cls):
    '''
    Descripttion: 获取语义分割的label，共19种，对应第一种online的方式
                  'background-0'、'skin-1', 'l_brow-2', 'r_brow-3', 'l_eye-4', 'r_eye-5',
                  'eye_g-6', 'l_ear-7', 'r_ear-8', 'ear_r-9',
                  'nose-10', 'mouth-11', 'u_lip-12', 'l_lip-13', 'neck-14',
                  'neck_l-15', 'cloth-16', 'hair-17', 'hat-18'

                  由语义分割的label合并得到自监督的label，共8种：
                  自监督label | 语义分割label
                  0-background: background    0
                  1-skin： skin   1
                  2-eye: l_brow + r_brow + l_eye + r_eye + eye_g  2,3,4,5,6
                  3-ear: l_ear + r_ear + ear_r    7,8,9
                  4-nose: nose    10
                  5-mouth: mouth + u_lip + l_lip  11,12,13
                  6-neck: neck + neck_l + cloth   14,15,16
                  7-hair： hair + hat 17,18
                  将像素最多的前n个设定为1，其余设定为0

    Param: 
    Return: 
    '''
    
    
    cls_label=get_cls_labels_from19(seg_label, topk_cls)
    return cls_label

def get_cls_labels_from19(seg_lable, topk_cls):
    cls_label_19=seg_lable
    count_19=[]
    for i in range(19):
        count_19.append(sum(sum(cls_label_19==i)))
    
    # print('count_19: ', count_19)
    # ch_id=[[0],[1],[2,3,4,5,6],[7,8,9],[10],[11,12,13],[14,15,16],[17,18]]
    # cls_label_8=[]
    # for ids in ch_id:
    #     sum_=0
    #     for id in ids:
    #         sum_+=count_19[id]
    #     cls_label_8.append(sum_)
    # topk_cls=np.random.randint(1,topk_cls,1)[0]
    # print('topk_cls: ',topk_cls)
    _,pre=torch.tensor(count_19).topk(topk_cls,dim=0,largest=True) #get pos of topk labels
    # print('cls_label_8: ', cls_label_8)
    cls_label=torch.zeros((1,19))
    for pos in pre:
        if count_19[pos]!=0:
            cls_label[0][pos]=1
    # print(cls_label)
    return cls_label.squeeze(0)

def get_cls_labels_from8(seg_lable, topk_cls):
    cls_label_8=seg_lable
    count_8=[]
    for i in range(8):
        count_8.append(sum(sum(cls_label_8==i)))
    
    # print('count_8: ', count_8)
    # ch_id=[[0],[1],[2,3,4,5,6],[7,8,9],[10],[11,12,13],[14,15,16],[17,18]]
    # cls_label_8=[]
    # for ids in ch_id:
    #     sum_=0
    #     for id in ids:
    #         sum_+=count_19[id]
    #     cls_label_8.append(sum_)
    # topk_cls=np.random.randint(1,topk_cls,1)[0]
    # print('topk_cls: ',topk_cls)
    _,pre=torch.tensor(count_8).topk(topk_cls,dim=0,largest=True) #get pos of topk labels
    # print('cls_label_8: ', cls_label_8)
    cls_label=torch.zeros((1,8))
    for pos in pre:
        if count_8[pos]!=0:
            cls_label[0][pos]=1
    # print(cls_label)
    return cls_label.squeeze(0)

def get_train_img_attr(attr_file, img_dir, augment_dir, select_rate=1):
    with open(attr_file) as f:
        lines = f.readlines()
        set_size = len(lines)
        imgs = []
        attr = np.zeros((set_size, 40))

        for i, line in enumerate(lines):
            vals = line.split()
            imgs.append(vals[0])
            attr[i, :] = [vals[j + 1] for j in range(40)]
        # attr = np.tile(attr, (2, 1))
        # imgs = [os.path.join(img_dir, i) for i in imgs] + [os.path.join(augment_dir, i) for i in imgs]
        # imgs = [os.path.join(augment_dir, i) for i in imgs]
    return imgs[::select_rate], attr[::select_rate]

def get_test_img_attr(attr_file, img_dir, select_rate=1):
    with open(attr_file) as f:
        lines = f.readlines()
        set_size = len(lines)
        imgs = []
        attr = np.zeros((set_size, 40))

        for i, line in enumerate(lines):
            vals = line.split()
            imgs.append(vals[0])
            attr[i, :] = [vals[j + 1] for j in range(40)]
    return imgs[::select_rate], attr[::select_rate]

def random_rotate_mask(img, mask, rot):
    if rot==1:
        img=img.transpose(Image.ROTATE_90)
        mask=mask.transpose(Image.ROTATE_90)
    elif rot==2:
        img=img.transpose(Image.ROTATE_180)
        mask=mask.transpose(Image.ROTATE_180)
    elif rot==3:
        img=img.transpose(Image.ROTATE_270)
        mask=mask.transpose(Image.ROTATE_270)
    elif rot!=0:
        raise ValueError('rot error!')
    
    import matplotlib.pyplot as plt
    # print(rot)
    # plt.imshow(img);plt.show()
    # plt.pause(1)
    return img, mask

def random_rotate(img, rot):
    if rot==1:
        img=F.rotate(img, 90)
    elif rot==2:
        img=F.rotate(img, 180)
    elif rot==3:
        img=F.rotate(img, 270)
    elif rot!=0:
        raise ValueError('rot error!')

    return img