'''
Author: sy
Date: 2021-03-23 10:23:13
LastEditTime: 2021-11-01 14:36:47
LastEditors: your name

Description: 
- preact_resnet WITHOUT attention

# baseed on 
https://github.com/phuocphn/pytorch-imagenet-preactresnet/blob/master/models/preact_resnet.py


FilePath: /sy/s3net/models/preact_resnets.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(x))
        
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(out)
        else:
            shortcut = x
        
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, paddings=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        

    def forward(self, x):
        out = F.relu(self.bn1(x))
        input_out = out
        
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.conv3(out)

        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(input_out)
        else:
            shortcut = x

        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=40):
        super(PreActResNet, self).__init__()
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.bias.data.zero_()
        

    def _make_layer(self, block, plans, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  #在[stride]后接上数字，不是相加
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, plans, stride))
            self.in_planes = plans * block.expansion

        return nn.Sequential(*layers)
    

    def forward(self, x):
        out = self.conv1(x)
        # print("conv1:", out.shape)
        

 
        out = self.bn1(self.relu(out))
        out = self.maxpool(out)
        # print("maxpool:", out.shape)
        
        out = self.layer1(out)
        # print("layer1:", out.shape)

        out = self.layer2(out)
        # print("layer2:", out.shape)

        out = self.layer3(out)
        # print("layer3:", out.shape)

        out = self.layer4(out)
        # print("layer4:", out.shape)

        out = self.avgpool(out)
        # print("avgpool:", out.shape)
        
        out = out.view(out.size(0), -1)
        # print("view:", out.shape)
        

        out = self.fc(out)
        
        return out



def PreActResNet18(num_cls=40):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_cls)

def PreActResNet34():
    return PreActResNet(PreActBlock, [3, 4, 6, 3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3, 8, 36, 2])

def test():
    net = PreActResNet18()
    y = net((torch.randn(1, 3, 224, 224)))
    # print(net)
    print(y, y.size())

if __name__ == '__main__':
    test()


        

        
    
    