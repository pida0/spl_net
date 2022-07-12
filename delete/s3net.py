'''
Author: sy
Date: 2021-03-26 09:01:55
LastEditors: pida0
LastEditTime: 2022-07-12 16:43:56
Description: 
'''
import torch.nn as nn
import torch
from torch.nn import functional as F
from .attention import RCCAModule, GLAtt


def conv_bn(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True))
    
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
    def __init__(self, num_classes_rot, num_classes_seg, num_classes_cls, block, num_blocks):
        super(s3net, self).__init__()
        
        self.preprocess = nn.ModuleList([
            conv_bn(27,3),
            conv_bn(9, 3),
            conv_bn(9, 3),
            conv_bn(9,3)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        
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
            conv_bn(64* 3, 64),
            conv_bn(128* 3, 128),
            conv_bn(256* 3, 256),
            conv_bn(512* 3, 512)
            # nn.Conv2d(in_channels=64* 3, out_channels=64, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(in_channels=128* 3, out_channels=128, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(in_channels=256* 3, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.Conv2d(in_channels=512* 3, out_channels=512, kernel_size=1, stride=1, padding=0)
        ])

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # PCT分类层
        self.pct_cls = nn.ModuleList([
            nn.Linear(512*block.expansion, num_classes_cls),
            nn.Linear(512*block.expansion, num_classes_cls),
            nn.Linear(512*block.expansion, num_classes_cls),
            nn.Linear(512*block.expansion, num_classes_cls)
        ])
        
        # PRT分类层
        self.fusion = conv_bn(512 * block.expansion * 3, 512 * block.expansion)
        self.rot_cls = nn.Linear(512 * block.expansion, num_classes_rot)

        # PST分类层, x = self.seg_cls(x, self.recurrence), 直接得到特征图
        # 前向传播的时候使用插值法扩大面积
        # F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
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
    
    
    def forward(self, x_psct, x_prtg, x_prtl_):
        '''
        @description: 
        @param {x_psct：pst和pct使用的random patch；
                x_prtg：prt中使用的聚合的global块，通道数为27=9*3（通道维度合并）；
                x_prtl：prt中使用的聚合的local块，是一个tensor（先通道维度合并得到c=9，然后三个堆叠），里面包含了上中下三个local块，每个块的通道数为9=3*3
                图像大小均为224*224}
        @return {out_prt_id：PRT的旋转id预测，[B, 9]
                 out_pst_mask：PST的语义分割mask，[B, 8, 7, 7]
                 out_pct_att：PCT的关键部位预测，一个list，长度为4（4个分支的结果），每个元素大小为[B, 2]，目前合并了，大小为[B, 8]
                 x_prtg_fake, x_prtg_real：PRT中local分支合成的伪全局特征和global分支的真全局特征，[B, 512, 7, 7]}
        '''
        # =================== head
        x_prtg = self.preprocess[0](x_prtg)
        x_prtg = self.head(x_prtg)
        
        x_prtl_ = x_prtl_.permute(1, 0, 2, 3, 4)
        x_prtl = [0] * 3
        for i in range(3):
            x_prtl[i] = self.head(self.preprocess[i+1](x_prtl_[i]))
        x_psct = self.head(x_psct)

        # ===================== 中间层
        out_prt = [x_prtg, x_prtl[0], x_prtl[1], x_prtl[2]]
        out_psct = [x_psct, x_psct, x_psct, x_psct]
        # i表示使用的block块序号
        for i in range(4):
            # j表示branch序号
            # preact block
            for j in range(4):
                out_prt[j] = self.branches[j][i * 2](out_prt[j])
                out_psct[j] = self.branches[j][i * 2](out_psct[j])
            
            concat_prtl = torch.cat((out_prt[1], out_prt[2], out_prt[3]), 1)
            concat_psct = torch.cat((out_psct[1], out_psct[2], out_psct[3]), 1)
            concat_prtl = self.aggregate[i](concat_prtl)
            concat_psct = self.aggregate[i](concat_psct)

            out_prt_g = out_prt[0]
            out_psct_g = out_psct[0]

            # attention block i
            for j in range(4):
                if j==0: # global branch
                    out_prt[j] = self.branches[j][i * 2 + 1](out_prt[j], concat_prtl)
                    out_psct[j] = self.branches[j][i * 2 + 1](out_psct[j], concat_psct)
                else: # local branch
                    out_prt[j] = self.branches[j][i * 2 + 1](out_prt[j], out_prt[0]) #out_prt_g-->out_prt[0]
                    out_psct[j] = self.branches[j][i * 2 + 1](out_psct[j], out_psct[0]) # 同上修改
            

            

        # ================== PRT 分类
        x_prtg_real = out_prt[0]
        x_prtg_real = self.gap(x_prtg_real)
        x_prtg_real = x_prtg_real.view(x_prtg_real.size(0), -1)

        out_prtl = torch.cat((out_prt[1], out_prt[2], out_prt[3]), 1)
        out_prtl = self.fusion(out_prtl)
        out_prtl = self.gap(out_prtl)
        out_prtl = out_prtl.view(out_prtl.size(0), -1)
        
        x_prtg_fake = out_prtl
        
        out_prt_id = self.rot_cls(out_prtl)
        
        # =================== PCT & PST 分类
        
        # print(out_psct[2].shape)
        out_pst_mask = self.seg_cls(out_psct[0], 2)

        out_pct_att = [0] * 4
        for i in range(4):
            out_psct[i] = self.gap(out_psct[i])
            out_psct[i] = out_psct[i].view(out_psct[i].size(0), -1)
            out_pct_att[i] = self.pct_cls[i](out_psct[i])
        
        # out_pct_att=torch.cat(out_pct_att,dim=1)

        

        return out_prt_id, out_pst_mask, out_pct_att, x_prtg_fake, x_prtg_real


class Discriminators(nn.Module):
    def __init__(self, in_dim=512):
        super(Discriminators, self).__init__()

        layers = []
        drop_p = 0.2
        out_dim = 64
        lk = 0.05

        for _ in range(3):
            
            layers += [nn.Linear(in_dim, out_dim, bias=False)]
            # layers += [nn.BatchNorm1d(out_dim, affine=True)]
            layers += [nn.LeakyReLU(lk, inplace=True)]
            # layers += [nn.Dropout(p=drop_p)]

            in_dim = out_dim
            out_dim = in_dim * 2
            # drop_p += 0.1
            
        layers += [nn.Linear(in_dim, 1, bias=False)]
        self.fc_adv = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    
    def forward(self, x):
        x = self.fc_adv(x)
        return x
        
def get_model(num_classes_rot, num_classes_seg, num_classes_cls, block=PreActBlock, num_blocks=[2,2,2,2]):
    return s3net(num_classes_rot, num_classes_seg, num_classes_cls, block, num_blocks)
    


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

   

        