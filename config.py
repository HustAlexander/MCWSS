class Defaultconfig(object):
    # class variable,every instance will share it
    data_root1 = '/home/kz/dataset/BratsTrain_2D_2019/HGG'
    data_root2 = '/home/kz/dataset/BratsTrain_2D_2019/LGG'
    batch_size = 16
    use_gpu = True
    max_epoch = 40
    lr = 0.00001
    lr_decay = 0.95
    weight_decay = 0.0001

opt=Defaultconfig
