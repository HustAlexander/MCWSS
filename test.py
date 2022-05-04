from config import opt
import torch
import dataset
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import weight_init,C_Net
from model_cross import weight_init,C_Net_
import numpy as np
from evaluation import *
#from logger import Logger
from torch.utils.data import DataLoader
import os
import nibabel as nib
from PIL import Image
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "0"



export_dir='./out'
data_root = '/home/kz/dataset/MICCAI_BraTS_2019_Data_Validation/'
# data_root = './dataset/ISLES2018_Training/TRAINING'


def normalize( data, smooth=1e-9):
    mean = data.mean()
    std = data.std()
    if (mean == 0) or (std == 0):
        return data
    # mean = data[data!=0].mean()
    # std = data[data!=0].std()
    data = (data - mean + smooth) / (std + smooth)
    # data[data==data.min()] = -10  # data.min() are 0 before normalization
    return data


with torch.no_grad():
    model=C_Net(img_ch=4,feature_ch=64)
    state_dict = torch.load('./models/cl.ckpt')
    model.load_state_dict(state_dict)
    model =model.cuda().eval()

    data_p = os.listdir(data_root)
    # data_p = sorted(data_p, key=lambda x: (int(x.split('_')[-2])))

    for folder in data_p:
        if 'BraTS' in folder:
            for nii in os.listdir(os.path.join(data_root,folder)):
                if '_t1.' in nii and 'nii' in nii:
                    t1_p=os.path.join(data_root,folder, nii)
                elif '_t1ce.' in nii and 'nii' in nii:
                    t1ce_p=os.path.join(data_root,folder, nii)
                elif '_t2.' in nii and 'nii' in nii:
                    t2_p=os.path.join(data_root,folder, nii)
                elif '_flair.' in nii and 'nii' in nii:
                    flair_p=os.path.join(data_root,folder, nii)



                
            for i in range(np.size(nib.load(t1_p).dataobj[:,:,:],2)):
                t1 = np.expand_dims(normalize(nib.load(t1_p).dataobj[:, :, i]),0)
                t1ce = np.expand_dims(normalize(nib.load(t1ce_p).dataobj[:, :, i]),0)
                t2 = np.expand_dims(normalize(nib.load(t2_p).dataobj[:, :, i]),0)
                flair = np.expand_dims(normalize(nib.load(flair_p).dataobj[:, :, i]),0)


                input_ =  np.concatenate((t1,t1ce,t2,flair),axis = 0)
                input = np.expand_dims(input_, 0)

                # if i ==0:
                #     input = input_
                # else:
                #     input = np.concatenate((input,input_),0)

                input = torch.from_numpy(input).type(torch.FloatTensor)

                

                input = Variable(input.cuda())

                

                # output,output_ = model(input,input)

                # [cl11,cl21,cam11,cam21] = output
                # [cl_co11,cl_co21,cam_co11,cam_co21] = output_
                cl01,cl11,cl02,cl12,cam11,cam21 = model(input)
                # loss for segmentation
                SR1 = cam11[:,1,:]>0.5
                SR1 = SR1.int()
                SR2 = cam21[:,1,:]>0.5
                SR2 = (SR2.int()-SR1)>0
                SR2 = 2*SR2.int()

                SR = SR1+SR2

                if i ==0:
                    SR_ = SR
                else:
                    SR_ = torch.cat((SR_,SR),0)

            SR = SR_.squeeze(1)

            SR = SR.cpu().numpy().transpose((1,2,0))

            T1 = nib.load(t1_p)

            img = nib.Nifti1Image(SR, T1.affine, T1.header)
            img.set_data_dtype(dtype=np.ushort)

            




            id = t1_p.split('.')[-3].split('/')[-1].split('_t1')[-2]

            export_path = os.path.join(export_dir, id + '.nii.gz')
            img.to_filename(export_path)






