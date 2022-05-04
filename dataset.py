import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T
import nibabel as nib
import h5py
import random




class create_cross(data.Dataset):
    def __init__(self, data_root1, data_root2,transforms=None, train=True, test=False):
        self.test = test

    
        datas1 = [os.path.join(data_root1, png) for png in os.listdir(data_root1)]
        datas2 = [os.path.join(data_root2, png) for png in os.listdir(data_root2)]

        datas1 = sorted(datas1, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )
        datas2 = sorted(datas2, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                     int(x.split('_')[-1].split('.')[-2])) )

        
        self.train =train
        if self.train:
            self.datas1 = datas1[int(0.3* len(datas1)):int(1* len(datas1)):3]
            self.datas2 = datas2[int(0.3* len(datas2)):int(1* len(datas2)):3]

        elif self.test:
            self.datas1 = datas1[int(0.9 * len(datas1)):]
            self.datas2 = datas2[int(0.9 * len(datas2)):]
        else:
            self.datas1 = datas1[int(0* len(datas1)):int(0.3 * len(datas1)):2]
            self.datas2 = datas2[int(0* len(datas2)):int(0.3 * len(datas2)):2]

        
        self.datas = self.datas1+self.datas2


        self.transforms1 = T.RandomRotation(90)
        
        w = np.loadtxt('Label_cross.txt',dtype=int)
        self.w = torch.from_numpy(w)




    def __getitem__(self, index):

        

        input,label1,label2,label3,label4,Mask1,Mask2,Mask3,Mask4 = self.read(index)
        label = torch.cat((label1.unsqueeze(0)+label4.unsqueeze(0),label2.unsqueeze(0)),0).unsqueeze(0).long()
        label = (label>0).long()
        cross = torch.matmul(label,self.w)
        cross = cross.numpy()

        if self.train:
            if cross.sum() > 0:
                index_=random.choice(np.where(cross>0)[1])
            else:
                index_ = index
        else:
            index_ = index
        
        

        input_,label1_,label2_,label3_,label4_,Mask1_,Mask2_,Mask3_,Mask4_ = self.read(index_)
        input = torch.cat((input,input_),0)
        label1 = torch.cat((label1.unsqueeze(0),label1_.unsqueeze(0)),0)
        label2 = torch.cat((label2.unsqueeze(0),label2_.unsqueeze(0)),0)
        label4 = torch.cat((label4.unsqueeze(0),label4_.unsqueeze(0)),0)



        return input, label1, label2,label4,Mask1+Mask4,Mask1+Mask2+Mask4,self.datas[index]
        

    def read(self,index):
        f = h5py.File(self.datas[index],'r')
        t1 =  np.expand_dims(f['t1'][:],0)
        t1ce =  np.expand_dims(f['t1ce'][:],0)
        t2 =  np.expand_dims(f['t2'][:],0)
        flair =  np.expand_dims(f['flair'][:],0)
        gt =  np.expand_dims(f['label'][:],0)

        t1 = self.normalize(t1)
        t1ce = self.normalize(t1ce)
        t2 = self.normalize(t2)
        flair = self.normalize(flair)

        input = np.concatenate((t1,t1ce,t2,flair),axis = 0)




        if self.train:
            data = np.concatenate((input,gt),axis = 0)
            data = torch.from_numpy(data).type(torch.FloatTensor)
            data = self.transforms1(data)
            

            input = data[:4,:,:]
            gt = data[4,:,:].unsqueeze(0)    
        else:
            input = torch.from_numpy(input).type(torch.FloatTensor)
            gt = torch.from_numpy(gt*1.0).type(torch.FloatTensor)  
        
        Mask1 = (gt==1).int()
        Mask2 = (gt==2).int()
        Mask3 = (gt==3).int()
        Mask4 = (gt==4).int()

        label1 = Mask1.sum()>0 
        label2 = Mask2.sum()>0 
        label3 = Mask3.sum()>0 
        label4 = Mask4.sum()>0 


        return input,label1,label2,label3,label4,Mask1,Mask2,Mask3,Mask4

    def normalize(self, data, smooth=1e-9):
        mean = data.mean()
        std = data.std()
        if (mean == 0) or (std == 0):
            return data

        data = (data - mean + smooth) / (std + smooth)
        return data


    def __len__(self):

        return len(self.datas)

