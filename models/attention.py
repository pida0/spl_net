'''
Author: sy
Date: 2021-03-23 15:59:17
LastEditors: pida0
LastEditTime: 2022-07-12 16:40:51
Description: criss-cross attention module and other attention blocks

cc_attention is borrowed from Serge-weihao/CCNet-Pure-Pytorch
GLAtt is borrowed from mtan & delian
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class GLAtt(nn.Module):
    def __init__(self, first_c, second_c, is_last=True):
        super(GLAtt, self).__init__()

        if first_c != second_c:
            self.pre_layer = nn.Sequential(
                nn.BatchNorm2d(first_c),
                nn.PReLU(),
                nn.Conv2d(in_channels=first_c, out_channels=second_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(second_c),
                nn.PReLU(),
            )

        
        self.first_layer = nn.Sequential(
            nn.BatchNorm2d(second_c),
            nn.PReLU(),
            nn.Conv2d(in_channels=second_c, out_channels=second_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(second_c),
            nn.Sigmoid()
        )

        if is_last:
            self.second_layer = nn.Sequential(
                nn.BatchNorm2d(second_c),
                nn.PReLU(),
                nn.Conv2d(in_channels=second_c, out_channels=second_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(second_c),
                nn.PReLU()
            )

        
    def forward(self, origin_f, att_f):

        if hasattr(self, 'pre_layer'):
            origin_f = self.pre_layer(origin_f)
            origin_f = F.max_pool2d(origin_f, kernel_size=3, stride=2, padding=1)

        att_f = self.first_layer(att_f)
        att_f = origin_f * att_f

        if hasattr(self,'second_layer'):
            att_f = self.second_layer(att_f)
        
        
        return att_f

class CrossAtt(nn.Module):
    def __init__(self, first_c, second_c, reduction=16):
        super(CrossAtt, self).__init__()

        if first_c != second_c:
            self.pre_layer = nn.Sequential(
                nn.BatchNorm2d(first_c),
                nn.PReLU(),
                nn.Conv2d(in_channels=first_c, out_channels=second_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(second_c),
                nn.PReLU(),
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att_layer = nn.Sequential(
            nn.Linear(second_c*2, second_c*2 // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(second_c*2 // reduction, second_c*2, bias=False),
            nn.Sigmoid()
        )
        self.downsample_layer = nn.Sequential(
            nn.BatchNorm2d(second_c*2),
            nn.PReLU(),
            nn.Conv2d(in_channels=second_c*2, out_channels=second_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(second_c),
            nn.PReLU()
        )


        self.spatial_att_layer = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


        
    def forward(self, origin_f, att_f):

        if hasattr(self, 'pre_layer'):
            origin_f = self.pre_layer(origin_f)
            origin_f = F.max_pool2d(origin_f, kernel_size=3, stride=2, padding=1)

        channel_f = torch.cat([origin_f, att_f], dim=1) 
        spatial_f = origin_f + att_f 
        

        b, c, _, _ = channel_f.size()
        y = self.avg_pool(channel_f).view(b, c)
        y = self.channel_att_layer(y).view(b, c, 1, 1)
        out_channel_f = channel_f + channel_f * y.expand_as(channel_f) 
        out_channel_f = self.downsample_layer(out_channel_f) 

        spatial_avg_f = torch.mean(spatial_f, dim=1, keepdim=True)
        spatial_max_f, _ = torch.max(spatial_f, dim=1, keepdim=True)
        y = torch.cat([spatial_avg_f, spatial_max_f], dim=1)
        y = self.spatial_att_layer(y)
        out_spatial_f = spatial_f + spatial_f * y 
        
        out_f = out_channel_f + out_spatial_f 

        
        return out_f

if __name__ == '__main__':
   
    model=RCCAModule(512,256,8)
    # model = GLAtt(64,128)
    x = torch.randn(2, 512, 7, 7)
    # y = torch.randn(2, 64, 56, 56)
    x = x.cuda()
    # y = y.cuda()
    model=model.cuda()
    out = model(x)
    print(out.shape)
