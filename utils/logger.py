'''
Author: sy
Date: 2021-03-31 15:49:16
LastEditors: your name
LastEditTime: 2022-07-01 02:05:03
Description: file content
'''

import torch
import numpy as np


from .seg_metric import SegmentationMetric
from .cls_metric import cal_attr_acc, compute_accuracy


def record(prt_out, prt_angle_out, pst_out, pct_out, labels, writer, iter_counter, mode):
    prec1, prec5 = compute_accuracy(prt_out.cpu().data, labels[2].cpu().data, topk=(1, 2))
    rot_acc = prec1.item()

    a_prec1, a_prec5 = compute_accuracy(prt_angle_out.cpu().data, labels[3].cpu().data, topk=(1, 2))
    angle_acc = a_prec1.item()

    correct = torch.FloatTensor(19).fill_(0)
    total = 0
    pred = torch.cat(pct_out, 1)
    label = torch.cat(labels[1], 1)
    c, t = cal_attr_acc(pred.cpu().data, label.cpu().data, 19)
    correct.add_(c)
    total += t
    self_cls_acc = torch.mean(correct/total)

    seg_metric = SegmentationMetric(19)
    seg_metric.addBatch(pst_out.cpu().data, labels[0].cpu().data)
    seg_acc = seg_metric.pixelAccuracy()
    seg_mIoU = seg_metric.meanIntersectionOverUnion()

    
    if mode =='train':        
        # writer.add_scalar('train_loss',loss.item(),iter_counter)
        writer.add_scalar('train_pct_acc',self_cls_acc,iter_counter)
        writer.add_scalar('train_prt_acc',rot_acc,iter_counter)
        writer.add_scalar('train_prt_angle_acc',angle_acc,iter_counter)
        writer.add_scalar('train_pst_acc',seg_acc,iter_counter)
        writer.add_scalar('train_pst_mIoU', seg_mIoU, iter_counter)

        print('[Acc ]: pst_acc: %.4f, pst_mIoU: %.4f, pct_acc: %.4f, prt_acc: %.4f, prt_angle_acc: %.4f' 
                % (seg_acc, seg_mIoU, self_cls_acc, rot_acc, angle_acc))
    
    return np.array([seg_acc, seg_mIoU, self_cls_acc.item(), rot_acc, angle_acc])

def show_eval_result(mean_acc, max_acc, max_epoch, cur_epoch, writer):
    
    for i in range(5):
        if max_acc[i] < mean_acc[i]:
            max_acc[i] = mean_acc[i]
            max_epoch[i] = cur_epoch
        
        
        
        
    print('| Test epoch %d, pst_acc: %.4f, pst_mIou: %.4f, pct_acc: %.4f, prt_acc: %.4f, prt_angle_acc: %.4f'
               % (cur_epoch, mean_acc[0], mean_acc[1], mean_acc[2], mean_acc[3], mean_acc[4]))
    print('| Current max acc and its epoch: pst_acc: %.4f@ep%d, pst_mIou: %.4f@ep%d, pct_acc: %.4f@ep%d, prt_acc: %.4f@ep%d, prt_angle_acc: %.4f@ep%d' %
           (max_acc[0], max_epoch[0], max_acc[1], max_epoch[1], max_acc[2], max_epoch[2], max_acc[3], max_epoch[3], max_acc[4], max_epoch[4]))

    writer.add_scalar('eval_prt_acc', mean_acc[3], cur_epoch)
    writer.add_scalar('eval_prt_angle_acc', mean_acc[4], cur_epoch)
    writer.add_scalar('eval_pct_acc', mean_acc[2], cur_epoch)
    writer.add_scalar('eval_pst_acc', mean_acc[0], cur_epoch)
    writer.add_scalar('eval_pst_mIoU', mean_acc[1], cur_epoch)
    