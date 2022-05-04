# SAWSL
Semantic Affinity-Aware Weakly Supervised Learning for Multi-Class Medical Image Segmentation with Slice-Level Labels 
This code is a simple implemention example of on the BraTS2019 dataset.
1. Transform your dataset from 3D nii scans to 2D h5 slices with nii_h5.py, or you can recode the dataset.py
2. Run the group.py to obtain the ratio between the positive and negtive samples.
3. Train the model.
4. Test your model.
It should be noted that this framework just generate the class activation maps, and you can use the CAMs as pseudo labels to train a segmentation network e.g. U-Net, further improve the segmentation performance.  
