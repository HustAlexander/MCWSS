from binascii import a2b_hex
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T
import nibabel as nib
import h5py



data_root = '/home/kz/dataset/BratsTrain_2D_2019/HGG'



datas = [os.path.join(data_root, png) for png in os.listdir(data_root)]
# targets = [os.path.join(target_root, png) for png in os.listdir(target_root)]

datas = sorted(datas, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                int(x.split('_')[-1].split('.')[-2])) )


datas = datas[int(0.3* len(datas)):int(0.7* len(datas)):3]
print(len(datas))

a = torch.zeros(2,len(datas))

index = 0
for data in datas:
    f = h5py.File(data)
    gt =  np.expand_dims(f['label'][:],0)
    gt = torch.from_numpy(gt*1.0).type(torch.FloatTensor) 

    Mask1 = (gt==1).int()
    Mask2 = (gt==2).int()
    # Mask3 = (gt==3).int()
    Mask4 = (gt==4).int()

    label1 = (Mask1.sum()>0).float()
    label2 = (Mask2.sum()>0).float()
    # label3 = (Mask3.sum()>0).float()
    label4 = (Mask4.sum()>0).float()

    a[0,index] = ((label1+label4)>0).float()
    a[1,index] = label2



    index+=1


a = a.to(torch.uint8)
print(a.size())
a = a.numpy()
np.savetxt('./Label_cross.txt',a,fmt='%d')


        




