'''
Author: sy
Date: 2021-03-26 09:01:55
LastEditors: pida0
LastEditTime: 2022-07-12 16:45:37
Description: 
'''
import torch.nn as nn
import torch
from torch.nn import functional as F
from .attention import RCCAModule, GLAtt, CrossAtt

def conv_bn(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_dim), nn.PReLU())

class PreActBlock(nn.Module):
    '''
    @description: 在resnet18和resnet34中用到的block
    @param {*}
    @return {*}
    '''    

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1.cuda()
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
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.prelu1 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = nn.PReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.prelu3 = nn.PReLU()
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
    
    def forward(self, x):
        out = self.prelu1(self.bn1(x))
        input_out = out

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv2(out)
        out = self.bn3(out)

        out = self.prelu3(out)
        out = self.conv3(out)

        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(input_out)
        else:
            shortcut = x

        out = out + shortcut
        return out



class s3net(nn.Module):
    def __init__(self, num_classes_cls, block, num_blocks, dataset='CelebA'):
        '''
        @description: 
        @param {num_classes_cls： 一个包含4个元素的list，表示每个分支学习到的属性个数}
        @return {*}
        '''
        super(s3net, self).__init__()
        
        self.dataset=dataset
        self.cls_num=num_classes_cls


        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.branches = [0] * 5
        for i in range(5):
            self.in_planes = 64
            if i == 0:  
                self.branches[i] = nn.ModuleList([
                    self._make_layer(block, 64, num_blocks[0], stride=1),
                    self._make_layer(block, 128, num_blocks[1], stride=2),
                    self._make_layer(block, 256, num_blocks[2], stride=2),
                    self._make_layer(block, 512, num_blocks[3], stride=2)
                ])
            else:
                self.branches[i] = nn.ModuleList([
                    CrossAtt(64*block.expansion, 64*block.expansion),
                    CrossAtt(64*block.expansion, 64*block.expansion*2),
                    CrossAtt(64*block.expansion*2, 64*block.expansion*4),
                    CrossAtt(64*block.expansion*4, 64*block.expansion*8)
                ])
        self.branches = nn.Sequential(*self.branches)
        
        

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # PCT分类层
        if dataset == 'MAAD':
            self.pct_cls = nn.ModuleList([
                nn.Linear(512*block.expansion, num_classes_cls[0]*3),
                nn.Linear(512*block.expansion, num_classes_cls[1]*3),
                nn.Linear(512*block.expansion, num_classes_cls[2]*3),
                nn.Linear(512*block.expansion, num_classes_cls[3]*3)
            ])
        else:
            self.pct_cls = nn.ModuleList([
                nn.Linear(512*block.expansion, num_classes_cls[0]),
                nn.Linear(512*block.expansion, num_classes_cls[1]),
                nn.Linear(512*block.expansion, num_classes_cls[2]),
                nn.Linear(512*block.expansion, num_classes_cls[3])
            ])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def _make_layer(self, block, plans, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  #在[stride]后接上数字，不是相加
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, plans, stride))
            self.in_planes = plans * block.expansion

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        '''
        @description: 
        @param {x: 人脸图像}
        @return {out_att：属性预测，一个list，长度为4（4个分支的结果），每个元素大小为[B, num_classes_cls[i]]}
        '''
        # =================== head
        x = self.head(x)
        # ===================== 中间层
        out = [0] * 5
        # i表示使用的block块序号
        for i in range(4):
            # j表示branch序号
            for j in range(5):
                if j == 0:  # 特征提取的branch
                    if i==0:
                        out[j] = self.branches[j][i](x)
                    else:
                        out[j] = self.branches[j][i](out[j])
                else: # 4个任务branch
                    if i == 0:  # 第一个block块，两个输入都是特征提取中的block输出
                        out[j] = self.branches[j][i](out[0], out[0])
                    else: # 后面的block块，两个输入分别是特征提取中的block输出和上一层自己的输出
                        out[j] = self.branches[j][i](out[j], out[0])
            

        # =================== attr 分类
        
        out_att = [0] * 4
        masks = [0] * 4
        for i in range(1,5):
            masks[i-1]=out[i]
            out[i] = self.gap(out[i])
            out[i] = out[i].view(out[i].size(0), -1)
            out_att[i-1] = self.pct_cls[i-1](out[i])
            if self.dataset=='MAAD':
                out_att[i-1]=out_att[i-1].reshape(-1,self.cls_num[i-1],3)
        
        return out_att, masks
    
    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        # print(param_dict.keys())
        # print(self.state_dict().keys())
        for i in self.state_dict():
            # if 'cls' in i or 'fc' in i or 'last_layer' in i:
            if 'cls' in i or 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict['module.' + i])
    
    def load_param_multi_gpu(self, model_path):
        param_dict = torch.load(model_path)
        # print(param_dict.keys())
        for i in self.state_dict():
            # if 'cls' in i or 'fc' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict['module.' + i])
        
def get_down_model_5b(num_classes_cls=[12, 13, 6, 9], block=PreActBlock, num_blocks=[2, 2, 2, 2], dataset='CelebA'):
    return s3net(num_classes_cls, block, num_blocks,dataset)
    


if __name__ == '__main__':
    model = get_model(9, 8, 2)

    # print(model)
    x = torch.randn(16, 3, 224, 224)
    y = torch.randn(16, 27, 224, 224)
    z = torch.randn(16, 9, 224, 224)
    z = torch.stack((z, z, z))
    
    x = x.cuda()
    y = y.cuda()
    z = z.cuda()
    print(z.size())
    model=model.cuda()
    prt, pst, pct, x_prtg_fake, x_prtg_real= model(x,y,z)
    print(prt.shape,pst.shape,pct.shape, x_prtg_fake.shape,x_prtg_real.shape)

   

        