#!/usr/bin/python3
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18, resnet34, resnet50


from lib.utils import load_model




""" fusion two level features """
class BasicConv2d(nn.Module):    #很多模块的使用卷积层都是以其为基础，论文中的BConvN
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, norm_layer=nn.BatchNorm2d):
        super(FAM, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(256)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left_mask.size()[2:]:  
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')    
            z1 = F.relu(left_mask * down_, inplace=True)    
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)                 

        out = torch.cat((z1, z2), dim=1)         
        return F.relu(self.bn3(self.conv3(out)), inplace=True) 

class CAM(nn.Module):
    def  __init__(self,input_channel,output_channel,norm_layer = nn.BatchNorm2d):
        super(CAM,self).__init__()
        self.conva = nn.Conv2d(input_channel,256,kernel_size=3,stride=1,padding=1)
        self.convb = nn.Conv2d(input_channel,256,kernel_size=3,stride=1,padding=1)
        self.c = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.weights = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(input_channel, input_channel, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU()
        )
        self.bn4 = norm_layer(256)
    def forward(self,first,second):
        first_mask = self.conva(first)
        second_mask = self.convb(second)
        if first_mask.size()[2:] != second_mask.size()[2:]:
            first_down =F.interpolate(first_mask,second_mask.size()[2:],mode='bilinear')

            w = self.weights(first_down)
            w = F.interpolate(w, size=first_down.size()[2:], mode='bilinear', align_corners=True)
            x_cat = self.c( w )

            return  F.relu(self.bn4(self.d(x_cat * second_mask)),inplace=True)
            #return  F.relu(self.bn4(torch.mul(first_down,second_mask)),inplace=True)
        else:
            #return  F.relu(self.bn4(torch.mul(first_mask,second_mask)),inplace=True)
            w = self.weights(first_mask)
            w = F.interpolate(w, size=first_mask.size()[2:], mode='bilinear', align_corners=True)
            x_cat = self.c(w)

            return F.relu(self.bn4(self.d(x_cat * second_mask)), inplace=True)
'''        
        if first_mask.size()[2:] != second.size()[2:]:
            second_down = F.interpolate(second,first_mask.size()[2:],mode='bilinear')
            z1 = F.relu(first_mask*second_down,inplace=True)
        else:
            z1 = F.relu(first_mask*second,inplace=True)
        if first.size()[2:] != second_mask.size()[2:]:
            second_mask = F.interpolate(second_mask,first.size()[2:],mode='bilinear')
        z2 = F.relu(second_mask*first,inplace=True)
        out = torch.cat((z1,z2),dim=1)
        return  F.relu(self.bn4(self.convc(out)),inplace=True)
        
        if first_mask.size()[2:] != second_mask.size()[2:]:
            first_down = F.interpolate(first_mask,size=second_mask.size()[2:],mode='bilinear')
            #out = torch.cat((first_down,second_mask),dim=1)
            #return F.relu(self.bn4(torch.mul(first_down,second_mask)))
            return F.relu(self.bn4(self.convc(torch.cat((first_down,second_mask),dim=1))),inplace=True)
        else:
            return F.relu(self.bn4(self.convc(torch.cat(first_mask,second_mask))),inplace=True)

class DAM (nn.Module):
    def __int__(self,first_channel,second_channel,third_channel,norm_layer = nn.BatchNorm2d):
        super(DAM,self).__int__()
        self.conva = nn.Conv2d(first_channel,256,kernel_size=3,stride=1,padding=1)
        self.convb = nn.Conv2d(second_channel,256,kernel_size=3,stride=1,padding=1)
        self.convc = nn.Conv2d(third_channel,256,kernel_size=3,stride=1,padding=1)
        self.convd = nn.Conv2d(256*3,256,kernel_size=3,stride=1,padding=1)
        self.bn4 = norm_layer(256)
    def forward(self,first,second,third):
        first_mask = self.conva(first)
        second_mask = self.convb(second)
        third_mask = self.convc(third)
        if second_mask.size()[2:] != first_mask.size()[2:] or third_mask.size()[2:] != first_mask.size()[2:]:
            second_down = F.interpolate(second_mask,size=first_mask.size()[2:],mode='bilinear')
            third_down = F.interpolate(third_mask,size=first_mask.size()[2:],mode='bilinear')
            #out = torch.cat((first_mask,second_down,third_down),dim=1)
            return F.relu(self.bn4(self.convd(torch.cat((first_mask,second_down,third_mask),dim=1))),inplace=True)
        else:
            return F.relu(self.bn4(self.convd(torch.cat((first_mask,second_mask,third_mask),dim=1))),inplace=True)
'''




class CCA(nn.Module):          
    def __init__(self, in_channel, ratio, out_channel):
        super(CCA, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(out_channel, out_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(out_channel// ratio, out_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = max_out + avg_out
        x= x * self.sigmoid(out)

        return x

class SA(nn.Module):    #空间注意力机制，先将通道数转化为1
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        avg_out = torch.mean(x2, dim=1, keepdim=True)
        x2 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x2)
        return self.sigmoid(x2) * x1

class CMAT(nn.Module):
    def __init__(self, in_channel, ratio, CA=True):
        super(CMAT, self).__init__()
        self.CA = CA
        self.CCA1 = CCA(in_channel, ratio, out_channel=256)
        self.CCA2 = CCA(in_channel, ratio, out_channel=256)
        if self.CA:
            self.att1 = SA()
            self.att2 = SA()
        else:
            print("Warning: not use CrossAttention!")
            self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
            self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)



    def forward(self, rgb, depth, beta, gamma,gate):
        rgb = self.CCA1(rgb)                
        depth = self.CCA2(depth)
        if self.CA:                            
            feat_1 = self.att1(rgb, depth)
            feat_2 = self.att2(depth, rgb)
        else:
            w1 = self.conv2(rgb)
            w2 = self.conv3(depth)          
            feat_1 = F.relu(w2*rgb, inplace=True)
            feat_2 = F.relu(w1*depth, inplace=True)

        out1 = rgb + gate * beta * feat_1
        out2 = depth + gamma * feat_2      

        return out1, out2


class Fusion(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):       
        super(Fusion, self).__init__()
        self.conv0 = nn.Conv2d(in_channel*2, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)

    def forward(self, x1, x2, alpha,beta):
        out1 = alpha * x1 + beta*(1.0 - alpha) * x2
        out2 = x1 * x2
        out  = torch.cat((out1, out2), dim=1)
        out = F.relu(self.bn0(self.conv0(out)), inplace=True)    

        return out

class side_fusion(nn.Module):
    def __init__(self, in_channel, norm_layer =nn.BatchNorm2d):
        super(side_fusion,self).__init__()
        self.conv0 = nn.Conv2d(in_channel*2, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)
    def forward(self,sideout1,sideout2):
        out = torch.cat((sideout1,sideout2),dim=1)
        out = F.relu(self.bn0(self.conv0(out)),inplace=True)

        return out

class global_fusion(nn.Module):
    def __init__(self,in_channel,out_channel,norm_layer = nn.BatchNorm2d):
        super(global_fusion,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,256,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(out_channel,256,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(256*2,256,kernel_size=3,stride=1,padding=1)
        self.bn0 = norm_layer(256)
    def forward(self,gfusion1,gfusion2):
        gfusion1 = self.conv1(gfusion1)
        gfusion2 = self.conv2(gfusion2)
        if gfusion2.size()[2:] != gfusion1.size()[2:]:  #进行上采样
            gfusion2 = F.interpolate(gfusion2,gfusion1.size()[2:],mode='bilinear')
            global_fusion1 = torch.mul(gfusion1,gfusion2)
            out = F.relu(self.bn0(self.conv3(torch.cat((gfusion1,global_fusion1),dim =1 ))))
            return out


class Segment(nn.Module):
    global backbone_rgb
    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm2d, cfg=None, aux_layers=True):
        super(Segment, self).__init__()
        self.cfg = cfg
        self.aux_layers = aux_layers

        if backbone == 'resnet18':
            channels = [64, 128, 256, 512]
            self.backbone_rgb = resnet18(in_channel=3, norm_layer=norm_layer)
            self.backbone_d = resnet18(in_channel=1, norm_layer=norm_layer)
            backbone_rgb = load_model(self.backbone_rgb, 'model_zoo/resnet18-5c106cde.pth')
            backbone_d = load_model(self.backbone_d, 'model_zoo/resnet18-5c106cde.pth', depth_input=True)
        elif backbone == 'resnet34':
            channels = [64, 128, 256, 512] # resnet34
            self.backbone_rgb =  resnet34(in_channel=3, norm_layer=norm_layer)
            self.backbone_d = resnet34(in_channel=1, norm_layer=norm_layer)
            backbone_rgb = load_model(self.backbone_rgb, 'model_zoo/resnet34-333f7ec4.pth')
            backbone_d = load_model(self.backbone_d, 'model_zoo/resnet34-333f7ec4.pth', depth_input=True)
        elif backbone == 'resnet50':       #如果backbone是resnet50 下载模型参数
            channels = [256, 512, 1024, 2048]
            self.backbone_rgb = resnet50(in_channel=3, norm_layer=norm_layer)      #通过resnet文件返回resnet50的模型 return model
            self.backbone_d = resnet50(in_channel=1, norm_layer=norm_layer)
            backbone_rgb= load_model(self.backbone_rgb, './model_zoo/resnet50-19c8e357.pth')   #下载预训练好的的参数
            print('backbone_rgb have download..')
            backbone_d = load_model(self.backbone_d, './model_zoo/resnet50-19c8e357.pth', depth_input=True)
            print('backbone_d have download...')
        else:
            raise Exception("backbone:%s does not support!"%backbone)
        if backbone_rgb is None:
            print("Warning: the model_zoo of {} does no exist!".format(backbone))
        if backbone_d is None:
            print("Warning: the model_zoo of {} does no exist!".format(backbone))
        else:
            self.backbone_rgb = backbone_rgb
            self.backbone_d = backbone_d
            print('have')

        # fusion modules
        self.cmat5 = CMAT(channels[3], 64, True)     #2048
        self.cmat4 = CMAT(channels[2], 64, True)     #1024
        self.cmat3 = CMAT(channels[1], 64, True)     #512
        self.cmat2 = CMAT(channels[0], 64, True)     #256

        # low-level & high-level
        self.fam54_1 = FAM(256, 256)                        
        self.fam43_1 = FAM(256, 256)
        self.fam32_1 = FAM(256, 256)
        self.fam54_2 = FAM(256, 256)
        self.fam43_2 = FAM(256, 256)
        self.fam32_2 = FAM(256, 256)
         #GMA 输出融合
        self.cam1 = CAM(256,64)
        self.cam2 = CAM(256,64)
        self.cam3 = CAM(256,64)
        self.cam4 = CAM(256,64)
         #三个GMA输出融合
        #self.dam1 = DAM(256,256,256)
        #self.dam2 = DAM(256,256,256)

        # fusion, TBD
        self.fusion = Fusion(256)

        self.sidefusion1 = side_fusion(256)
        self.sidefusion2 = side_fusion(256)

        self.gfusion1 = global_fusion(256,256)
        self.gfusion2 = global_fusion(256,256)
        self.gfusion3 = global_fusion(256,256)
        self.gfusion4 = global_fusion(256,256)

        self.sigmoid = nn.Sigmoid()
        #生成fuzhutu canshu
        if self.aux_layers:
            # self.linear5_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            # self.linear4_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            # self.linear3_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            # self.linear2_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            # self.linear5_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            # self.linear4_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            # self.linear3_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            # self.linear2_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.side_fusion3 = nn.Conv2d(256,1,kernel_size=3,stride=1,padding=1)
            self.side_fusion4 = nn.Conv2d(256,1,kernel_size=3,stride=1,padding=1)
        self.linear_out = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)           
        self.fc = nn.Sequential(                         
                   nn.Linear(channels[-1]*2, 512),
                   ##nn.Dropout(p=0.3),
                   nn.ReLU(True),
                   nn.Linear(512, 256+1),
                   nn.Sigmoid(),
                   )

        self.initialize()

    def forward(self, rgb, depth):
        raw_size = rgb.size()[2:]
        bz = rgb.shape[0]
        enc2_1, enc3_1, enc4_1, enc5_1 = self.backbone_rgb(rgb) 
        enc2_2, enc3_2, enc4_2, enc5_2 = self.backbone_d(depth)

        rgb_gap = self.gap1(enc5_1)
        rgb_gap = rgb_gap.view(bz, -1)
        depth_gap = self.gap2(enc5_2)                        
        depth_gap = depth_gap.view(bz, -1)
        feat = torch.cat((rgb_gap, depth_gap), dim=1)
        feat = self.fc(feat)

        gate = feat[:, -1].view(bz, 1, 1, 1)

        alpha = feat[:, :256]                                  
        alpha = alpha.view(bz, 256, 1, 1)


        out5_1, out5_2 = self.cmat5(enc5_1, enc5_2,1,1,gate)
        fir4_1, fir4_2   = self.cmat4(enc4_1, enc4_2,1,1, gate)  
        fir3_1, fir3_2   = self.cmat3(enc3_1, enc3_2,1,1, gate)
        fir2_1, fir2_2   = self.cmat2(enc2_1, enc2_2,1,1, gate)

        #三个GMA模块进行融合
        #de1 = self.dam1(fir2_1,fir3_1,fir4_1)
        #de2 = self.dam2(fir2_2,fir3_2,fir4_2)

        de4_1 = self.cam1(fir2_2,fir4_1)
        de4_2 = self.cam2(fir2_1,fir4_2)
        second3_1 = self.cam3(fir2_2,fir3_1)
        second3_2 = self.cam4(fir2_1,fir3_2)


        #自己添加的模块
        #de4_1 = torch.mul(de2_1,fir4_1)
        #de4_2 = torch.mul(de2_2,fir4_2)
        #de3_1 = torch.mul(de2_1,fir3_1)
        #de3_2 = torch.mul(de2_2,fir3_2)

        out4_1 = self.fam54_1(de4_1 , out5_1)
        out4_2 = self.fam54_2(de4_2, out5_2)
        side_fusion4 = self.sidefusion1(out4_1, out4_2)
        de3_1 = self.gfusion1(second3_1,side_fusion4)
        de3_2 = self.gfusion2(second3_2,side_fusion4)

        out3_1 = self.fam43_1(de3_1, out4_1)
        out3_2 = self.fam43_2(de3_2, out4_2)
        side_fusion3 =self.sidefusion2(out3_1,out3_2)
        de2_1 = self.gfusion3(fir2_1,side_fusion3)
        de2_2 = self.gfusion4(fir2_2,side_fusion3)
                                                        #F ronghe
        out2_1 = self.fam32_1(de2_1, out3_1)
        out2_2 = self.fam32_2(de2_2, out3_2)





        # final fusion
        out = self.fusion(out2_1, out2_2, alpha, gate)
        out = F.interpolate(self.linear_out(out), size=raw_size, mode='bilinear', )
        # aux_layer
        if self.training and self.aux_layers:
            # out5_1 = F.interpolate(self.linear5_1(out5_1), size=raw_size, mode='bilinear')   #双线性插值改变size
            # out4_1 = F.interpolate(self.linear4_1(out4_1), size=raw_size, mode='bilinear')
            # out3_1 = F.interpolate(self.linear3_1(out3_1), size=raw_size, mode='bilinear')
            # out2_1 = F.interpolate(self.linear2_1(out2_1), size=raw_size, mode='bilinear')
            # out5_2 = F.interpolate(self.linear5_2(out5_2), size=raw_size, mode='bilinear')
            # out4_2 = F.interpolate(self.linear4_2(out4_2), size=raw_size, mode='bilinear')
            # out3_2 = F.interpolate(self.linear3_2(out3_2), size=raw_size, mode='bilinear')
            # out2_2 = F.interpolate(self.linear2_2(out2_2), size=raw_size, mode='bilinear')
            side_fusion3 = F.interpolate(self.side_fusion3(side_fusion3),size=raw_size,mode='bilinear')
            side_fusion4 = F.interpolate(self.side_fusion4(side_fusion4),size=raw_size,mode='bilinear')

            return out, side_fusion3,side_fusion4,self.sigmoid(out),self.sigmoid(side_fusion3),self.sigmoid(side_fusion4),gate.view(bz, -1)
        else:

            return [out, gate.view(bz, -1)]

    def initialize(self):
        if self.cfg and self.cfg.snapshot:
            print("loading state dict:%s ..."%(self.cfg.snapshot))
            self.load_state_dict(torch.load(self.cfg.snapshot),strict=True)
        else:
            pass

