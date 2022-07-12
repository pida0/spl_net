'''
Author: sy
Date: 2021-03-26 09:01:55
LastEditors: pida0
LastEditTime: 2022-07-12 16:42:38
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
        self.prelu1=nn.PReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2=nn.PReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.prelu1(self.bn1(x))
        
        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(out)
        else:
            shortcut = x
        
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu2(out)
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



class s3net_5b(nn.Module):
    def __init__(self, num_classes_rot, num_classes_angle, num_classes_seg, num_classes_cls, block, num_blocks):
        super(s3net_5b, self).__init__()
        
        self.preprocess = conv_bn(3*num_classes_rot, 3)
       

        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        
        
        
        # branches里面有四个分支，表示全局、上部、中部、下部
        # 每一个分支里面有四个preact block和四个对应的attention block
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
        self.pct_cls = nn.ModuleList([
            nn.Linear(512*block.expansion, num_classes_cls[0]),
            nn.Linear(512*block.expansion, num_classes_cls[1]),
            nn.Linear(512*block.expansion, num_classes_cls[2]),
            nn.Linear(512*block.expansion, num_classes_cls[3])
        ])
        
        # PRT分类层
        self.fusion = conv_bn(512 * block.expansion * 3, 512 * block.expansion)
        self.rot_cls = nn.Linear(512 * block.expansion, num_classes_rot)
        self.angle_cls = nn.Linear(512 * block.expansion, num_classes_angle)

        # PST分类层,
        self.seg_cls = RCCAModule(512, 256, num_classes_seg)


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
    
    
    def forward(self, x_psct, x_prtg):
        
        # =================== head
        x_prtg = self.preprocess(x_prtg)
        x_prtg = self.head(x_prtg)
        


        x_psct = self.head(x_psct)

        # ===================== 中间层
        
        out_prt = [0] * 5
        out_psct = [0] * 5
        # i表示使用的block块序号
        for i in range(4):
            # j表示branch序号
            for j in range(5):
                if j == 0:  # 特征提取的branch
                    if i==0:
                        out_prt[j] = self.branches[j][i](x_prtg)
                        out_psct[j] = self.branches[j][i](x_psct)
                    else:
                        out_prt[j] = self.branches[j][i](out_prt[j])
                        out_psct[j] = self.branches[j][i](out_psct[j])
                else: # 4个任务branch
                    if i == 0:  # 第一个block块，两个输入都是特征提取中的block输出
                        out_psct[j] = self.branches[j][i](out_psct[0], out_psct[0])
                    else: # 后面的block块，两个输入分别是特征提取中的block输出和上一层自己的输出
                        out_psct[j] = self.branches[j][i](out_psct[j], out_psct[0])
            
                        
            
        # ================== PRT 分类
       

        out_prt_f = out_prt[0]
        out_prt_f = self.gap(out_prt_f)
        out_prt_f = out_prt_f.view(out_prt_f.size(0), -1)
        out_prt_id = self.rot_cls(out_prt_f)
        out_prt_angle = self.angle_cls(out_prt_f)
        
        # =================== PCT & PST 分类

        x_psct_real = out_psct[1]
        x_psct_real = self.gap(x_psct_real)
        x_psct_real = x_psct_real.view(x_psct_real.size(0), -1)

        out_psctl = torch.cat((out_psct[2], out_psct[3], out_psct[4]), 1)
        out_psctl = self.fusion(out_psctl)
        out_psctl = self.gap(out_psctl)
        out_psctl = out_psctl.view(out_psctl.size(0), -1)
        x_psct_fake = out_psctl
        
        # print(out_psct[2].shape)
        out_pst_mask = self.seg_cls(out_psct[0], 2)

        out_pct_att = [0] * 4
        for i in range(1,5):
            out_psct[i] = self.gap(out_psct[i])
            out_psct[i] = out_psct[i].view(out_psct[i].size(0), -1)
            out_pct_att[i-1] = self.pct_cls[i-1](out_psct[i])

        


        

        return out_prt_id, out_prt_angle, out_pst_mask, out_pct_att, x_psct_fake, x_psct_real
    
    def load_param_multi_gpu(self, model_path):
        param_dict = torch.load(model_path)
        # print(param_dict.keys())
        for i in self.state_dict():
            # if 'cls' in i or 'fc' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict['module.' + i])


class Discriminators(nn.Module):
    def __init__(self, in_dim=512):
        super(Discriminators, self).__init__()

        layers = []
        drop_p = 0.2
        out_dim = 64
        lk = 0.05

        for _ in range(3):
            
            layers += [nn.Linear(in_dim, out_dim, bias=False)]
            layers += [nn.LeakyReLU(lk, inplace=True)]

            in_dim = out_dim
            out_dim = in_dim * 2
            # drop_p += 0.1
            
        layers += [nn.Linear(in_dim, 1, bias=False)]
        self.fc_adv = nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.fc_adv(x)
        return x


class ClsNet(nn.Module):
    def __init__(self, in_dim=512, num_classes_rot=9):
        super(ClsNet, self).__init__()

        self.fc_adv = nn.Linear(in_dim, num_classes_rot)
    
    
    def forward(self, x):
        x = self.fc_adv(x)
        return x

def get_model_5b(num_classes_rot, num_classes_seg, num_classes_cls=[2,7,4,6], block=PreActBlock, num_blocks=[2,2,2,2]):
    return s3net_5b(num_classes_rot, 4, num_classes_seg, num_classes_cls, block, num_blocks)
    


if __name__ == '__main__':
    model = get_model_5b(16,19)

    # print(model)
    x = torch.randn(24, 3, 75, 75)
    y = torch.randn(24, 27, 75, 75)
    z = torch.randn(16, 9, 224, 224)
    z = torch.stack((z, z, z))
    
    x = x.cuda()
    y = y.cuda()
    z = z.cuda()
    print(z.size())
    model=model.cuda()
    prt, pst, pct, x_prtg_fake, x_prtg_real= model(x,y)
    print(prt.shape,pst.shape,pct.shape, x_prtg_fake.shape,x_prtg_real.shape)

   

        