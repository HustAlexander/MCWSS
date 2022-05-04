import glob
import os
import h5py
import numpy as np
import SimpleITK as sitk
import random
from tqdm import tqdm
from skimage import measure



slice_num = 0
cnt = 0

data_root= './BraTS_2019_Data_Training/HGG/'

train_list = os.listdir(data_root) 
i = 0

for case in tqdm(train_list):
    i+=1
    name = []
    print(f'name：{case}')
    
    t1_path = os.path.join(data_root, case,case+'_t1.nii.gz')
    t1ce_path = os.path.join(data_root, case,case+'_t1ce.nii.gz')
    t2_path = os.path.join(data_root, case,case+'_t2.nii.gz')
    flair_path = os.path.join(data_root,case,case+'_flair.nii.gz')

    msk_path = os.path.join(data_root, case,case+'_seg.nii.gz')

    t1_itk = sitk.ReadImage(t1_path)
    t1 = sitk.GetArrayFromImage(t1_itk)
    t1 = t1.astype(np.int16)
    
    t1ce_itk = sitk.ReadImage(t1ce_path)
    t1ce = sitk.GetArrayFromImage(t1ce_itk)
    t1ce = t1ce.astype(np.int16)

    t2_itk = sitk.ReadImage(t2_path)
    t2 = sitk.GetArrayFromImage(t2_itk)
    t2 = t2.astype(np.int16)

    flair_itk = sitk.ReadImage(flair_path)
    flair = sitk.GetArrayFromImage(flair_itk)
    flair = flair.astype(np.int16)



    msk_itk = sitk.ReadImage(msk_path)
    mask = sitk.GetArrayFromImage(msk_itk)



    for slice_ind in range(t1.shape[0]):
        f = h5py.File(
            './BratsTrain_2D_2019/HGG/{}_slice_{}.h5'.format(i, slice_ind), 'w')
        f.create_dataset(
            't1', data=t1[slice_ind], compression='gzip')
        f.create_dataset(
            't1ce', data=t1ce[slice_ind], compression='gzip')
        f.create_dataset(
            't2', data=t2[slice_ind], compression='gzip')
        f.create_dataset(
            'flair', data=flair[slice_ind], compression='gzip')
        f.create_dataset('label', data=mask[slice_ind], compression='gzip')
        # f.create_dataset(
        #     'lung_mask', data=lung_mask[slice_ind], compression='gzip')
        f.close()

        name.append(f'{case}_slice_{slice_ind}')
        slice_num += 1


print('Converted all covid-2019 volumes to 2D slices')
print('Total {} slices'.format(slice_num))
