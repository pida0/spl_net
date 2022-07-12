'''
Author: sy
Date: 2021-03-26 09:01:55
LastEditors: pida0
LastEditTime: 2022-07-12 16:46:50
Description: s3net-spatial semantic split net for FAR
             总共四条分支：global、upper、middle、lower，四条分支之间使用attention机制交互。
             其中，global、upper、middle、lower共同输出图像的40个属性，每个分支的输出与PSMCNN保持一致
             
             与s3net相比，去掉了对抗学习、语义分割、旋转预测、global分支的cc-attention
'''
import torch.nn as nn
import torch
from torch.nn import functional as F
from ..models.attention import RCCAModule, GLAtt

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

class s3net(nn.Module):
    def __init__(self, num_classes_cls, block, num_blocks):
        '''
        @description: 
        @param {num_classes_cls： 一个包含4个元素的list，表示每个分支学习到的属性个数}
        @return {*}
        '''
        super(s3net, self).__init__()
        
        # # 用来处理PRT的输入，直接将几个patch通道维度拼接，然后减少通道数
        # self.preprocess = nn.ModuleList([
        #     nn.Conv2d(in_channels=27, out_channels=3, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0)
        # ])

        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 这里还没有修改，只是照搬preact-resnet18，可能会造成语义分割的特征图变小
        # 可以尝试ccnet的做法，使用空洞卷积，但是这样又要考虑在为PRT和PCT服务时候特征图太大的问题
        # 也可以直接上采样看看效果
        
        
        # branches里面有四个分支，表示全局、上部、中部、下部
        # 每一个分支里面有四个preact block和四个对应的attention block
        self.branches = [0] * 4
        for i in range(4):
            self.in_planes = 64
            self.branches[i] = nn.ModuleList([
                self._make_layer(block, 64, num_blocks[0], stride=1),
                GLAtt(64, 64),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                GLAtt(128, 128),
                self._make_layer(block, 256, num_blocks[2], stride=2),
                GLAtt(256, 256),
                self._make_layer(block, 512, num_blocks[3], stride=2),
                GLAtt(512, 512)
            ])
        self.branches = nn.Sequential(*self.branches)
        
        # 聚合local branch的卷积模块
        self.aggregate = nn.ModuleList([
            nn.Conv2d(in_channels=64* 3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=128* 3, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=256* 3, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=512* 3, out_channels=512, kernel_size=1, stride=1, padding=0)
        ])

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # PCT分类层
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
        out = [x, x, x, x]
        # i表示使用的block块序号
        for i in range(4):
            # j表示branch序号
            # preact block
            for j in range(4):
                out[j] = self.branches[j][i * 2](out[j])
            
            concat_local = torch.cat((out[1], out[2], out[3]), 1)
            concat_local = self.aggregate[i](concat_local)

            out_global = out[0]

            # attention block i
            for j in range(4):
                if j==0: # global branch
                    out[j] = self.branches[j][i * 2 + 1](out[j], concat_local)
                else: # local branch
                    out[j] = self.branches[j][i * 2 + 1](out[j], out_global)
            

        # =================== attr 分类
        
        out_att = [0] * 4
        for i in range(4):
            out[i] = self.gap(out[i])
            out[i] = out[i].view(out[i].size(0), -1)
            out_att[i] = self.pct_cls[i](out[i])
        
        # out_pct_att=torch.cat(out_pct_att,dim=1)
        return out_att
    
    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        # print(param_dict.keys())
        for i in self.state_dict():
            if 'cls' in i or 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict['module.'+i])
        
def get_down_model(num_classes_cls=[12, 13, 6, 9], block=PreActBlock, num_blocks=[2, 2, 2, 2]):
    return s3net(num_classes_cls, block, num_blocks)
    


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

   

        