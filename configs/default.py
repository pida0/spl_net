'''
Author: sy
Date: 2021-03-31 15:08:24
LastEditors: your name
LastEditTime: 2021-04-29 09:10:11
Description: file content
'''

from yacs.config import CfgNode as CN

_C=CN()

#============== model config ==================
# _C.MODEL=CN()





#============== data config ==================
# part_dir, attr_dir, partition, img_dir, mask_dir, type, topk_cls, select_rate=1
_C.DATA=CN()

_C.DATA.AUGMENT_DIR='/media/sy/disk/DataSet/LFW/lfw-deepfunneled/' # for lfwa
_C.DATA.PART_DIR='/media/sy/disk/DataSet/CelebA/Eval/list_eval_partition.txt'
_C.DATA.ATTR_DIR='/media/sy/disk/DataSet/CelebA/Anno/list_attr_celeba.txt'
_C.DATA.IMG_DIR='/media/sy/disk/DataSet/CelebA/img_align_celeba_crop75/'
_C.DATA.MASK_DIR='/media/sy/disk/DataSet/CelebA/crop75_mask/'
_C.DATA.TRAIN_FILE='/media/sy/disk/DataSet/LFW/anno/train.txt' # FOR LFWA
_C.DATA.TEST_FILE = '/media/sy/disk/DataSet/LFW/anno/test.txt'  #FOE LFWA
_C.DATA.TRAIN_IMG_DIR='/media/sy/disk/DataSet/LFW/anno/train.txt' # FOR MAAD
_C.DATA.TEST_IMG_DIR = '/media/sy/disk/DataSet/LFW/anno/test.txt'  #FOE MAAD
_C.DATA.TRAIN_PRT_IMG_DIR='/media/sy/disk/DataSet/CelebA/img_align_celeba/' # for maad
_C.DATA.TEST_PRT_IMG_DIR = '/media/sy/disk/DataSet/CelebA/img_align_celeba/'  # for maad
_C.DATA.TRAIN_MASK_DIR='/media/sy/disk/DataSet/CelebA/img_align_celeba/' # for maad
_C.DATA.TEST_MASK_DIR='/media/sy/disk/DataSet/CelebA/img_align_celeba/' # for maad
_C.DATA.PRT_IMG_DIR='/media/sy/disk/DataSet/CelebA/img_align_celeba/'
_C.DATA.TYPE='self-rot' # self | fine-tune | self-rot,for transform and rotation 
_C.DATA.PCT_TOPK=4
_C.DATA.PATCH_NUM=9
_C.DATA.SELECT_RATE=1
_C.DATA.DATASET_NAME='CelebA'


#============== train config ==================
_C.TRAIN=CN()
_C.TRAIN.BATCHSIZE=32
_C.TRAIN.N_EPOCH=60
_C.TRAIN.LR=0.0001
_C.TRAIN.IS_CONTINUE=False
_C.TRAIN.IS_PRETRAINED=True
_C.TRAIN.SAVE_PTH='/media/sy/disk/ckp/seg_ssl_v3/'
_C.TRAIN.LOAD_PTH='/media/sy/disk/ckp/seg_ssl_v3/'
_C.TRAIN.LOAD_EPOCH=0
_C.TRAIN.PRETRAIN_PTH='/media/sy/disk/ckp/seg_ssl_v3_pami/'
_C.TRAIN.PRETRAIN_EPOCH = 52
_C.TRAIN.D_REPEAT = 5


#============== test config ==================
_C.TEST=CN()
_C.TEST.BATCHSIZE=32
_C.TEST.LOAD_PTH='/media/sy/disk/ckp/seg_ssl_v3/'
_C.TEST.LOAD_EPOCH = 35



_C.SW_COMMENT='_seg_ssl_v3_test'
_C.PRINT_INTER=5
_C.SW_LOG_DIR='/media/sy/disk/runs/seg_ssl_v5_pr_celeba_nosmooth'