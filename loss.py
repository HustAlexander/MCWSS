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


        
        input_0 = input[:,0,:,:].view(input.size(0), -1)
        input_1 = input[:,1,:,:].view(input.size(0), -1)
        
        loss = torch.min(input_0, input_1).mean(dim=1) 

        
        label = label.view(-1)
        loss += label*input_0.mean(dim=1) + (1 - label) * input_1.mean(dim=1)

        
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





        cl2_ =  nn.functional.adaptive_avg_pool2d(cam2*(1-cam_1.detach()),(1,1))
        cl2_ = cl2_.view(-1, 2)

        cl1_ =  nn.functional.adaptive_avg_pool2d(cam1*(1-cam_2.detach()),(1,1))
        cl1_ = cl1_.view(-1, 2)

        loss1 = self.criterion(cl2_,label2)+0.1*self.criterion(cl1_,label1)

        cam_1 = cam_1.view(cam_1.size(0), -1)
        cam_2 = cam_2.view(cam_2.size(0), -1)
        label2 = label2.view(-1).unsqueeze(1)

        loss2 =label2*(torch.min(cam_1,cam_2).mean(dim=1)-0.1*torch.max(cam_1,cam_2))
        loss = loss1+0.1*loss2



        
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()  



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
