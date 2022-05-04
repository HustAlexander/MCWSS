import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            # nn.ConvTranspose2d(ch_in,ch_in,kernel_size=3,stride=2,padding=1),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		        nn.BatchNorm2d(ch_out),
			      nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x





class C_Net(nn.Module):
    def __init__(self, img_ch=4,feature_ch=16, output_ch=1):
        super(C_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(img_ch, feature_ch)
        self.Conv2 = conv_block(feature_ch, feature_ch*2)
        self.Conv3 = conv_block(feature_ch*2, feature_ch*4)
        self.Conv4 = conv_block(feature_ch*4, feature_ch*8)
        self.Conv5 = conv_block(feature_ch*8, feature_ch*16)
        self.Conv6 = conv_block(feature_ch*16, feature_ch*32)

        self.bott31 = nn.Conv2d(feature_ch*4, 2,kernel_size  = 1,bias = False)
        self.bott51 = nn.Conv2d(feature_ch*16, 2,kernel_size  = 1,bias = False)
        

        self.bott32 = nn.Conv2d(feature_ch*4, 2,kernel_size  = 1,bias = False)
        self.bott52 = nn.Conv2d(feature_ch*16, 2,kernel_size  = 1,bias = False)




        self.bott31_ = nn.Conv2d(feature_ch*4, 2,kernel_size  = 1,bias = False)
        self.bott51_ = nn.Conv2d(feature_ch*16, 2,kernel_size  = 1,bias = False)
        

        self.bott32_ = nn.Conv2d(feature_ch*4, 2,kernel_size  = 1,bias = False)
        self.bott52_ = nn.Conv2d(feature_ch*16, 2,kernel_size  = 1,bias = False)


        self.extra_linear_e3=nn.Linear(feature_ch*4, feature_ch*4,bias = False)

        self.extra_linear_e5 = nn.Linear(feature_ch*16, feature_ch*16,bias = False)


    ######## generate multi-level features #######
    def feature1(self,x):
        x1 = self.Conv1(x)       
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)                
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
    
        return x3,x5


    ######## generate decomposing features #######
    def co_att(self,x1,x2,extra_linear_e):
        fea_size1 = x1.size()[2:]
        all_dim1= fea_size1[0]*fea_size1[1]

        fea_size2 = x2.size()[2:]
        all_dim2= fea_size2[0]*fea_size2[1]

        x1_flat=x1.view(-1, x2.size()[1], all_dim1)
        x2_flat=x2.view(-1, x2.size()[1], all_dim2)

        x1_t = torch.transpose(x1_flat,1,2).contiguous()
        x1_corr = extra_linear_e(x1_t)

        A = torch.bmm(x1_corr, x2_flat)
        A = F.softmax(A, dim = 1)
        B = F.softmax(torch.transpose(A,1,2),dim=1)

        x2_att = torch.bmm(x1_flat, A).contiguous()
        x1_att = torch.bmm(x2_flat, B).contiguous()
        input1_att = x1_att.view(-1, x2.size()[1], fea_size1[0], fea_size1[1])  
        input2_att = x2_att.view(-1, x2.size()[1], fea_size2[0], fea_size2[1])

    
        return input1_att,input2_att
    

    ######## generate CAMs and classification score #######
    def CAM_G(self,x3,x5,bott3,bott5):
    

        cam0 = bott3(x3)
        cl0 = nn.functional.adaptive_avg_pool2d(cam0,(1,1))
        cl0 = cl0.view(-1, 2)

        
        cam0 = F.upsample(cam0, size=(120, 120), mode='bilinear')


        B, C, H, W = cam0.shape
        cam0 = cam0.view(B, -1)
        cam0 = cam0-cam0.min(dim=1, keepdim=True)[0]
        cam0 = cam0/(cam0.max(dim=1, keepdim=True)[0] + 1e-9)
        cam0 = cam0.view(B, C, H, W)

        cam1 = bott5(x5)
        cl1 = nn.functional.adaptive_avg_pool2d(cam1,(1,1))
        cl1 = cl1.view(-1, 2)

        
        cam1 = F.upsample(cam1, size=(120, 120), mode='bilinear')


        B, C, H, W = cam1.shape
        cam1 = cam1.view(B, -1)
        cam1 = cam1-cam1.min(dim=1, keepdim=True)[0]
        cam1 = cam1/(cam1.max(dim=1, keepdim=True)[0] + 1e-9)
        cam1 = cam1.view(B, C, H, W)
        
        cam = cam0*cam1

        cam = F.upsample(cam, size=(240, 240), mode='bilinear')


        return [cl1,cl0], cam   




    def forward(self, x1,x2):

        x31,x51 = self.feature1(x1)
        x32,x52 = self.feature1(x2)


        cl11,cam11 = self.CAM_G(x31,x51,self.bott31,self.bott51)
        cl21,cam21 = self.CAM_G(x31,x51,self.bott32,self.bott52)


        x_co31,x_co32=self.co_att(x31,x32,self.extra_linear_e3)
        x_co51,x_co52=self.co_att(x51,x52,self.extra_linear_e5)


        cl_co11,cam_co11 = self.CAM_G(x_co31,x_co51,self.bott31_,self.bott51_)
        cl_co21,cam_co21 = self.CAM_G(x_co31,x_co51,self.bott32_,self.bott52_)


        
        return [cl11,cl21,cam11,cam21],\
            [cl_co11,cl_co21,cam_co11,cam_co21]    

 

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



