import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CAMLoss(nn.Module):
    def __init__(self, size_average=True):
        super(CAMLoss,self).__init__()
        
        self.size_average = size_average
        self.epsilon = 1e-9
        
    def forward(self, input, label):
        #mask[mask>33/255]=0
        #mask[mask<5/255]=0
        #mask[mask>0]=1
        #mask =mask.float().squeeze(1).view(mask.size(0), -1)

        
        input_0 = input[:,0,:,:].view(input.size(0), -1)
        input_1 = input[:,1,:,:].view(input.size(0), -1)
        
        # loss = torch.min(input_0, input_1).mean(dim=1) 
        #loss += 1-torch.max(input_0, input_1).mean(dim=1) 
        
        label = label.view(-1)
        loss = label*input_0.mean(dim=1) + (1 - label) * input_1.mean(dim=1)
        #print(mask.size(),input_1.size())
        #loss += torch.mul((1-mask),input_1)
        
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum() 

class CAMLoss_s(nn.Module):
    def __init__(self, size_average=True):
        super(CAMLoss_s,self).__init__()
        
        self.size_average = size_average
        self.epsilon = 1e-9
        
    def forward(self, input0, input1,label):

        
        input0 = input0.view(input0.size(0), -1)
        input1 = input1.view(input1.size(0), -1)

        # loss = torch.min(input_0, input_1).mean(dim=1) 
        #loss += 1-torch.max(input_0, input_1).mean(dim=1) 
        
        label = label.view(-1)

        loss = (1 - label) * input0.mean(dim=1)+(1 - label) * input1.mean(dim=1)

        #print(mask.size(),input_1.size())
        #loss += torch.mul((1-mask),input_1)
        
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum() 
        
        
class exclusLoss(nn.Module):
    def __init__(self, size_average=True):
        super(exclusLoss,self).__init__()
        
        self.size_average = size_average
        self.epsilon = 1e-9
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, cam1,cam2,label1,label2):

    
        cam_1 = cam1[:,1,:,:].unsqueeze(1)
        cam_2 = cam2[:,1,:,:].unsqueeze(1)

        # cam_1 = cam_1.view(cam_1.size(0), -1)
        # cam_2 = cam_2.view(cam_2.size(0), -1)

        # cam_12 = cam_1>0.5
        # cam_21 = cam_2>0.5
        # cam_12 = cam_12.int().detach()
        # cam_21 = cam_21.int().detach()
        # loss1 = cam_12*cam_21*cam_1*cam_2

        # cam_12 = cam_12*(1-cam_21)
        # cam_21 = cam_21*(1-cam_12)
        # loss2 = -cam_12*torch.log(cam_1+1e-8)-cam_21*torch.log(cam_2+1e-8)
        # loss = loss1+0.1*loss2
        

        
        # cam = torch.cat((cam_1.unsqueeze(1),cam_2.unsqueeze(1)),1)


        # max, Mask = torch.max(cam, dim=1)
        # Mask1 = Mask==0
        # Mask2 = Mask==1
        # Mask1 = Mask1.int().detach()
        # Mask2 = Mask2.int().detach()

        # cam2 = cam_2
        # loss = Mask1*cam_2
        # loss -= Mask1*cam2[:,0,:,:].view(cam2.size(0), -1)



        cl2_ =  nn.functional.adaptive_avg_pool2d(cam2*(1-cam_1.detach()),(1,1))
        cl2_ = cl2_.view(-1, 2)

        cl1_ =  nn.functional.adaptive_avg_pool2d(cam1*(1-cam_2.detach()),(1,1))
        cl1_ = cl1_.view(-1, 2)

        loss1 = self.criterion(cl2_,label2)+0.1*self.criterion(cl1_,label1)

        cam_1 = cam_1.view(cam_1.size(0), -1)
        cam_2 = cam_2.view(cam_2.size(0), -1)
        label2 = label2.view(-1).unsqueeze(1)

        loss2 =label2*torch.min(cam_1,cam_2).mean(dim=1)
        loss = loss1+0.1*loss2

        # loss = torch.min(CL2[:,1], cam_2)
        #loss = cam_1*cam_2
        # loss = label*cam_1.detach()*cam_2



        
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()  

class exclusLoss_s(nn.Module):
    def __init__(self, size_average=True):
        super(exclusLoss_s,self).__init__()
        
        self.size_average = size_average
        self.epsilon = 1e-9
        
    def forward(self, cam1,cam2):

    
        cam_1 = cam1.view(cam_1.size(0), -1)
        cam_2 = cam2.view(cam_2.size(0), -1)

        loss = cam_1*cam_2

        
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()  


class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=0,alpha=0.7,reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        pt = input
        target = target.float()
        alpha = self.alpha

        loss = (- alpha * (1 - pt) ** self.gamma * target * torch.log(pt+1e-8) - \
                (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt+1e-8))

        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class QDLoss(torch.nn.Module):

    def __init__(self,reduction='elementwise_mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, weight1, weight2):

        tmp = torch.norm(weight1, 2, 0)
        feat_norm1 = weight1 / tmp.view(1, -1)
        tmp = torch.norm(weight2, 2, 0)
        feat_norm2 = weight2 / tmp.view(1, -1)

        x = feat_norm1 * feat_norm2
        x = torch.sum(x, 1)
        loss = torch.mean(torch.abs(x))

        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class AlignLoss(torch.nn.Module):

    def __init__(self,reduction='elementwise_mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, cam,cam_co,label):
        cam = cam[:,1,:,:]
        cam_co = cam_co[:,1,:,:]

        cam = cam.view(cam.size(0), -1)
        cam_co = cam_co.view(cam_co.size(0), -1)

        label = label.view(-1).unsqueeze(1)


        loss = -label*torch.min(cam,cam_co)


        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
