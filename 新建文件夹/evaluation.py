import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT>threshold
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    #print(corr,tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT>threshold

    # TP : True Positive
    # FN : False Negative
    # TP = ((SR==1)+(GT==1))==2
    # FN = ((SR==0)+(GT==1))==2
    TP = (SR==1) & (GT==1)
    FN = (SR==0) & (GT==1)
    #print('1',torch.sum(SR==1),torch.sum(GT==1))
    #print('tp,fn',torch.sum(TP),torch.sum(FN))
    # torch.sum() returns FloatTensor
    SE = (float(torch.sum(TP))+ 1e-6)/(float(torch.sum(TP+FN)) + 1e-6)
    # if needed
    # SE = (float(torch.sum(TP))+ 1e-6)/(float(torch.sum(TP)+torch.sum(FN)) + 1e-6)
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT>threshold

    # TN : True Negative
    # FP : False Positive
    TN = (SR==0)&(GT==0)
    FP = (SR==1)&(GT==0)

    SP = (float(torch.sum(TN))+ 1e-6)/(float(torch.sum(TN+FP)) + 1e-6)
    # SP = (float(torch.sum(TN))+ 1e-6)/(float(torch.sum(TN)+torch.sum(FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT>threshold

    # TP : True Positive
    # FP : False Positive
    TP = (SR==1) & (GT==1)
    FP = (SR==1) & (GT==0)

    PC = (float(torch.sum(TP))+ 1e-6)/(float(torch.sum(TP+FP)) + 1e-6)
    # PC = (float(torch.sum(TP))+ 1e-6)/(float(torch.sum(TP)+torch.sum(FP)) + 1e-6)
    #print('tp:',float(torch.sum(TP))+ 1e-6)
    #print('fp:',float(torch.sum(FP))+ 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = (2*SE*PC+ 1e-6)/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    #print(SR.sum())
    GT = GT>threshold
    
    #Inter = torch.sum((SR+GT)==2)
    #Union = torch.sum((SR+GT)>=1)
    Inter = torch.sum((SR==1) & (GT==1))
    Union = torch.sum((SR==1) | (GT==1))
    #print('Inter',Inter)
    #print('Union',Union)
    
    JS = (float(Inter)+ 1e-6)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT>threshold
    
    Inter = torch.sum((SR==1) & (GT==1))
    DC = (float(2*Inter)+ 1e-6)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    #print(SR.sum(),GT.sum(),Inter,float(torch.sum(SR)+torch.sum(GT)))
    return DC



