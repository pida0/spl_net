'''
Author: sy
Date: 2021-03-31 15:47:52
LastEditors: your name
LastEditTime: 2021-03-31 15:48:12
Description: file content
'''

import torch
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (btchasize,num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        # print(log_probs.size())
        # print(targets.unsqueeze(1).data.cpu().size())
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        # print(targets)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(1)
        # print(loss)
        loss=torch.mean(loss)
        # print(loss)
        
        return loss




import numpy as np


class SegmentationMetric(object):
    '''
    Descripttion: refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
                    confusionMetric
                    P\L     P    N
 
                    P      TP    FP
 
                    N      FN    TN
                    need to change the output of the model [B,C,W,H] to [B,W,H]

    Param: 
    Return: 
    '''
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc
 
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU
 
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # print(type(imgPredict[mask]))
        # print(type(imgLabel[mask]))
        label = self.numClass * imgLabel[mask] + torch.from_numpy(imgPredict[mask])
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
 
    def addBatch(self, imgPredict, imgLabel):
        imgPredict=imgPredict.numpy().argmax(1)
        # print(imgPredict.shape, imgLabel.shape)
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
 
 
if __name__ == '__main__':
    cri=CrossEntropyLabelSmooth(6)
    imgPredict = torch.rand((4,6,5,5))
    print(imgPredict)
    # parsing = imgPredict.cpu().numpy().argmax(1)
    # print(parsing)
    # print(parsing.shape)
    imgLabel = torch.randint(0,6,(4,5,5))
    print(imgLabel)
    metric = SegmentationMetric(6)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    loss=cri(imgPredict,imgLabel)
    print(loss)
    accs=0
    accs+=acc
    print(acc,type(mIoU))