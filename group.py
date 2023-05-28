from binascii import a2b_hex
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms as T
import nibabel as nib
import h5py



data_root1 = '/home/kz/dataset/BratsTrain_2D_2019/HGG'
data_root2 = '/home/kz/dataset/BratsTrain_2D_2019/LGG'



datas1 = [os.path.join(data_root1, png) for png in os.listdir(data_root1)]

datas1 = sorted(datas1, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                int(x.split('_')[-1].split('.')[-2])) )

datas2 = [os.path.join(data_root2, png) for png in os.listdir(data_root2)]

datas2 = sorted(datas2, key=lambda x: (int(x.split('_')[-3].split('/')[-1]),
                                int(x.split('_')[-1].split('.')[-2])) )

datas = datas1[int(0.3* len(datas1)):int(1* len(datas1)):3]+datas2[int(0.3* len(datas2)):int(1* len(datas2)):3]

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


        




