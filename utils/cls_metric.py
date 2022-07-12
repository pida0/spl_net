'''
Author: sy
Date: 2021-03-31 15:48:51
LastEditors: pida0
LastEditTime: 2022-07-12 16:46:13
Description: file content
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def confusion_out(out):
    SMALL=1e-16
    return - torch.mean(torch.log(F.softmax(out + SMALL, dim=-1)))

# p_logit: [batch, class_num]
# q_logit: [batch, class_num]
def kl_loss(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

def cal_attr_acc(output, labels, cls_num):
    correct = torch.FloatTensor(cls_num).fill_(0)
   
    com1=output.sigmoid()>0.5
    com2=labels.cpu().data>0
    # print(com1.shape,com2.shape)
    correct.add_((com1.eq(com2)).data.cpu().sum(0).type(torch.FloatTensor))
    return correct, labels.size(0)

class BalancedLoss(nn.Module):
    def __init__(self, num_classes=40):
        super(BalancedLoss, self).__init__()

        self.balance_attr_pos_prop = torch.FloatTensor([0.5] * num_classes) #40个0.5

    def forward(self, pred, target, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            multi class target(index) for each sample.
        """
        weights = torch.ones_like(pred).float()
        batch_size, class_size = target.shape
        batch_current_size = weights.sum(0).cpu() 
        # print('batch_current_size: ',batch_current_size)
        balance_num = self.balance_attr_pos_prop * batch_current_size
        # print('balance_num: ',balance_num)

        pos_sum = (target * weights).sum(0).cpu()
        neg_sum = batch_current_size - pos_sum
        pos_gt_idx = (pos_sum >= balance_num) 
        # print('pos_gt_idx: ', pos_gt_idx)
        neg_gt_idx = (neg_sum > balance_num)
        # print('neg_gt_idx: ', neg_gt_idx)
        # print(target[:, 0].numpy())
        # print(np.argwhere(target[:, 0].numpy() == pos_gt_idx[0].float().numpy()))

        # for i in hard_attr_idx:
        for i in range(class_size): #40
            majority_idx = np.array([j[0] for j in np.argwhere(target[:, i].cpu().numpy() == pos_gt_idx[i].float().numpy())])
            weights[majority_idx, i] *= balance_num[i] / len(majority_idx)

            minority_idx = np.array([j[0] for j in np.argwhere(target[:, i].cpu().numpy() == neg_gt_idx[i].float().numpy())])
            if len(minority_idx) > 0:
                weights[minority_idx, i] *= (batch_current_size[i] - balance_num[i]) / len(minority_idx)
        # target = target.cuda()

        loss = nn.BCEWithLogitsLoss(reduction="none")(pred, target) * weights
        return loss.mean()

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0) 

    _, pred = output.topk(maxk, 1, True, True)
   
    pred = pred.t()
   
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        # self.config = config
        self.gamma = 2
        self.alpha = 1
        self.size_average = True
        self.weight = None

    def forward(self, input, target):
        """

        :param input: the shape is
        :param target:
        :return:
        """
        if self.alpha is None or isinstance(self.alpha, (float, int)):
            self.alpha = (self.alpha * torch.ones((target.size(1), 2))).cuda()
        if isinstance(self.alpha, list):
            self.alpha = torch.stack((torch.tensor(self.alpha), 1 - torch.tensor(self.alpha)), dim=1).cuda()
        pt = torch.sigmoid(input).cuda()
        # loss = nn.BCELoss(reduction="none")(pt, target)
        loss = nn.BCEWithLogitsLoss(reduction="none")(input, target)
        loss = target * torch.pow(1 - pt, self.gamma) * loss + (1 - target) * torch.pow(pt, self.gamma) * loss

        if self.weight is not None:
            loss = loss * self.weight
        assert loss.shape == target.shape

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def set_weight(self, weight):
        self.weight = weight