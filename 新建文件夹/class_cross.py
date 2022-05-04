from config import opt
import torch
import dataset
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model_cross import weight_init,C_Net_
import numpy as np
from torch.utils.data import DataLoader
import os
from loss import CAMLoss,exclusLoss,QDLoss,AlignLoss
from torchvision import transforms as T
import random
from evaluation import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

random.seed(309)
np.random.seed(309)
torch.manual_seed(309)
torch.cuda.manual_seed_all(309)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




model=C_Net_(img_ch=4,feature_ch=64)
# state_dict = torch.load('./models/model_cl.ckpt')
# model.load_state_dict(state_dict)
#model = DeepLabV3()
#vgg_model = VGGNet(requires_grad=True)
#model=FCNs(pretrained_net=vgg_model, n_class=1)
# model.apply(weight_init)
# model.load_state_dict(torch.load('./models/AttU_Net.ckpt'))
model = model.cuda()

w = np.loadtxt('Label_cross.txt')
# w1 = (w[0,:]+w[3,:]>0).astype(int)
w1 = w[0,:].sum()/w.shape[1]
w2 = w[1,:].sum()/w.shape[1]
# w3 = w[2,:].sum()/w.shape[1]
# w4 = w[3,:].sum()/w.shape[1]

w1=torch.FloatTensor([w1,1-w1]).cuda()
w2=torch.FloatTensor([w2,1-w2]).cuda()
# w3=torch.FloatTensor([w3,1-w3]).cuda()
# w4=torch.FloatTensor([w4,1-w4]).cuda()

train_data = dataset.create_cross(opt.data_root1, opt.data_root2, train=True)
val_data = dataset.create_cross(opt.data_root1, opt.data_root2, train=False)
train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, 1)

lr = opt.lr
criterion1 = CAMLoss()
criterion3 = nn.L1Loss()
criterion4 = exclusLoss()
criterion5 = nn.CrossEntropyLoss(weight=w1)
criterion6 = nn.CrossEntropyLoss(weight=w2)
# criterion7 = nn.CrossEntropyLoss(weight=w3)
# criterion8 = nn.CrossEntropyLoss(weight=w4)
criterion9 = AlignLoss()
# optimizer = optim.RMSprop(model.parameters(), lr = opt.lr,alpha=0.9,weight_decay=opt.weight_decay)
optimizer=optim.Adam(model.parameters(),lr=opt.lr)

best_score = 0.6

flip = T.RandomVerticalFlip(p=1)

def compute_loss(cl1,cl2,cam1,cam2,label1,label2):
    loss1 = criterion1(cam2,label2)+2*criterion1(cam1,label1)
    loss2 = criterion6(cl2[0],label2)+criterion6(cl2[1],label2)+criterion5(cl1[0],label1)+criterion5(cl1[1],label1)
    loss3 = criterion4(cam1,cam2,label1,label2)
    loss = loss1+loss2+0.1*loss3

    return loss



for epoch in range(opt.max_epoch):
    model.train()
    l=0
    l1=0
    l2=0
    l3=0
    total=0
    correct=0

    acc = 0.  # Accuracy
    TP0 = 0.  
    FP0 = 0.  
    FN0 = 0.  
    TN0 = 0. 
    length = 0
    cl_correct1=0
    cl_correct2=0
    cl_correct3=0
    cl_correct4=0

    cl_correct1_ = 0
    cl_correct2_ = 0
 
    for i, (data, label1, label2,label4,Mask1,Mask2,data_name) in enumerate(train_dataloader):
        input = Variable(data.cuda())
        label1 = Variable(label1.cuda()).long()
        label2 = Variable(label2.cuda()).long()
        label4 = Variable(label4.cuda()).long()

        label1 = ((label1+label4)>0).long()


        label1_ = (label1[:,0] + label1[:,1]==2).long()
        label2_ = (label2[:,0] + label2[:,1]==2).long()

        output,output_ = model(input[:,:4,:],input[:,4:,:])

        [cl11,cl21,cam11,cam21] = output
        [cl_co11,cl_co21,cam_co11,cam_co21] = output_


        # bh = int(Cl1.size(0)/2)



        # loss1 for classificatiton
        _, predicted1 = torch.max(0.5*(cl11[0]+cl11[1]), 1)
        _, predicted2 = torch.max(0.5*(cl21[0]+cl21[1]), 1)

        # _, predicted12 = torch.max(0.5*(cl12[0]+cl12[1]), 1)
        # _, predicted22 = torch.max(0.5*(cl22[0]+cl22[1]), 1)


        _, predicted_co1 = torch.max(0.5*(cl_co11[0]+cl_co11[1]), 1)
        _, predicted_co2 = torch.max(0.5*(cl_co21[0]+cl_co21[1]), 1)

        # _, predicted_co12 = torch.max(0.5*(cl_co12[0]+cl_co12[1]), 1)
        # _, predicted_co22 = torch.max(0.5*(cl_co22[0]+cl_co22[1]), 1)


        # _, predicted4 = torch.max(Cl4, 1)

        loss1 = compute_loss(cl11,cl21,cam11,cam21,label1[:,0],label2[:,0])
        # loss2 = compute_loss(cl11,cl21,cam11,cam21)
        loss1_ = compute_loss(cl_co11,cl_co21,cam_co11,cam_co21,label1_,label2_)
        # loss2_ = compute_loss(cl11,cl21,cam11,cam21)
        loss_align = criterion9(cam11,cam_co11,label1_)+criterion9(cam21,cam_co21,label2_)


        l1 += loss1
        l2 += loss1_
        l3 += loss_align


        optimizer.zero_grad()
        (loss1+loss1_+0.1*loss_align).backward()
        optimizer.step()

        total+=label1.size(0)

        #if only one label of a picture in a batch is 1,what will happen
        label1 = label1>0
        cl_correct1+=(predicted1==label1[:,0]).sum().item()

        label2 = label2>0
        cl_correct2+=(predicted2==label2[:,0]).sum().item()

        label1_ = label1_>0
        cl_correct1_+=(predicted_co1==label1_).sum().item()

        label2_ = label2_>0
        cl_correct2_+=(predicted_co2==label2_).sum().item()


        # label4 = label4>0
        # cl_correct4+=(predicted4==label4).sum().item()
        


    Cl_Acc1=cl_correct1/total
    Cl_Acc2=cl_correct2/total
    Cl_Acc1_=cl_correct1_/total
    Cl_Acc2_=cl_correct2_/total
    # Cl_Acc4=cl_correct4/total


    

    if (epoch + 1) % 20==0 and epoch + 1>=20:
        lr = lr*0.9
        print('reset learning rate to:', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    print(
        'Epoch [%d/%d], Loss: %.4f, %.4f, %.4f, \n[Training] Cl_Acc1: %.4f,Cl_Acc2: %.4f, Cl_Acc1_: %.4f,Cl_Acc2_: %.4f,\n' % (
            epoch + 1, opt.max_epoch,l1,l2,l3,
            Cl_Acc1,Cl_Acc2,Cl_Acc1_,Cl_Acc2_))

            


    model.eval()
    l = 0
    l1 =0
    l2 =0
    total=0
    correct=0
    cl_correct1=0
    cl_correct2=0
    cl_correct3=0
    cl_correct4=0
    cl_correct1_=0
    cl_correct2_=0
    
    acc_c = 0.  # Accuracy
    SE_c = 0.  # Sensitivity (Recall)
    SP_c = 0.  # Specificity
    PC_c = 0.  # Precision
    DC_c = 0.  # Dice Coefficient
    acc = 0.  # Accuracy
    SE = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    DC = 0.  # Dice Coefficient

    id = 0
    s = 1
    p = 0

    with torch.no_grad():
        for i, (data, label1, label2,label4,Mask1,Mask2,data_name) in enumerate(val_dataloader):
            input = Variable(data.cuda())
            GT_1 = Variable(Mask1.cuda())
            GT_2 = Variable(Mask2.cuda())
            label1 = Variable(label1.cuda()).long()
            label2 = Variable(label2.cuda()).long()
            label4 = Variable(label4.cuda()).long()

            label1 = ((label1+label4)>0).long()

            
            
            label1_ = (label1[:,0] + label1[:,1]==2).long()
            label2_ = (label2[:,0] + label2[:,1]==2).long()
        

            output,output_ = model(input[:,:4,:],input[:,4:,:])

            

            [cl11,cl21,cam11,cam21] = output
            [cl_co11,cl_co21,cam_co11,cam_co21] = output_

            SR1_c = cam11[:,1,:,:].unsqueeze(1)
            SR2_c = cam21[:,1,:,:].unsqueeze(1)


            # bh = int(Cl1.size(0)/2)



            # loss1 for classificatiton
            _, predicted1 = torch.max(0.5*(cl11[0]+cl11[1]), 1)
            _, predicted2 = torch.max(0.5*(cl21[0]+cl21[1]), 1)

            # _, predicted12 = torch.max(0.5*(cl12[0]+cl12[1]), 1)
            # _, predicted22 = torch.max(0.5*(cl22[0]+cl22[1]), 1)


            _, predicted_co1 = torch.max(0.5*(cl_co11[0]+cl_co11[1]), 1)
            _, predicted_co2 = torch.max(0.5*(cl_co21[0]+cl_co21[1]), 1)

            # _, predicted_co12 = torch.max(0.5*(cl_co12[0]+cl_co12[1]), 1)
            # _, predicted_co22 = torch.max(0.5*(cl_co22[0]+cl_co22[1]), 1)


            # _, predicted4 = torch.max(Cl4, 1)

            loss1 = compute_loss(cl11,cl21,cam11,cam21,label1[:,0],label2[:,0])
            # loss2 = compute_loss(cl11,cl21,cam11,cam21)
            loss1_ = compute_loss(cl_co11,cl_co21,cam_co11,cam_co21,label1_,label2_)
            # loss2_ = compute_loss(cl11,cl21,cam11,cam21)
            # loss_align = criterion9(cam11,cam_co11,label1_)+criterion9(cam21,cam_co21,label2_)


            l1 += loss1
            l2 += loss1_
            # l3 += loss_align


            total+=label1.size(0)

            #if only one label of a picture in a batch is 1,what will happen
            label1 = label1>0
            cl_correct1+=(predicted1==label1[:,0]).sum().item()

            label2 = label2>0
            cl_correct2+=(predicted2==label2[:,0]).sum().item()

            label1_ = label1_>0
            cl_correct1_+=(predicted_co1==label1_).sum().item()

            label2_ = label2_>0
            cl_correct2_+=(predicted_co2==label2_).sum().item()

            SR1_c = SR1_c>0.5
            SR2_c = SR2_c>0.5
            SR2_c = SR1_c+SR2_c

        
            if data_name[0].split('_')[-3] != id:

                if p > 0:
                    if torch.sum(GT2) != 0:
                        acc_c += get_accuracy(SR1, GT1)
                        SE_c += get_sensitivity(SR1, GT1)
                        SP_c += get_specificity(SR1, GT1)
                        DC_c += get_DC(SR1, GT1)

                        acc += get_accuracy(SR2, GT2)
                        SE += get_sensitivity(SR2, GT2)
                        SP += get_specificity(SR2, GT2)
                        DC += get_DC(SR2, GT2)


                id = data_name[0].split('_')[-3]
                SR1 = SR1_c
                GT1 = GT_1
                SR2 = SR2_c
                GT2 = GT_2
                s = 1
                p += 1
            else:
                SR1 = torch.cat((SR1,SR1_c),1)
                GT1 = torch.cat((GT1,GT_1),1)

                SR2 = torch.cat((SR2,SR2_c),1)
                GT2 = torch.cat((GT2,GT_2),1)
                s += 1

        ###last scan###  
        if torch.sum(GT2) != 0:
            acc_c += get_accuracy(SR1, GT1)
            SE_c += get_sensitivity(SR1, GT1)
            SP_c += get_specificity(SR1, GT1)
            DC_c += get_DC(SR1, GT1)

            acc += get_accuracy(SR2, GT2)
            SE += get_sensitivity(SR2, GT2)
            SP += get_specificity(SR2, GT2)
            DC += get_DC(SR2, GT2)



            


        Cl_Acc1=cl_correct1/total
        Cl_Acc2=cl_correct2/total
        Cl_Acc1_=cl_correct1_/total
        Cl_Acc2_=cl_correct2_/total
        # Cl_Acc4=cl_correct4/total
        

        score =  DC/ p
        # score =  Cl_Acc2
        
        

        print(
            '[val] Loss: %.4f, %.4f, \n Cl_Acc1: %.4f,Cl_Acc2: %.4f, Cl_Acc1_: %.4f,Cl_Acc2_: %.4f,\n' % (
                 l1,l2,
                 Cl_Acc1,Cl_Acc2,Cl_Acc1_,Cl_Acc2_))

        print(
            '[Seg]  Acc: %.4f, %.4f, SE: %.4f, %.4f, SP: %.4f, %.4f, DC: %.4f, %.4f\n'  % (
                acc_c/ p, acc/ p, SE_c/ p, SE/ p, SP_c/ p,SP/ p, DC_c/ p,DC/ p))

        # Save Best model
        if score > best_score:
            best_score = score
            best_Net = model.state_dict()
            print('Best model score : %.4f \n' % (best_score))
            torch.save(best_Net, './models/model_cl_cross_align_OL.ckpt')













