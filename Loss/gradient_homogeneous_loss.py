import torch
import torch.nn.functional as F

import numpy as np

def pred_compute_layer_indices(img_tensor,mode):
   
    img_padded = F.pad(img_tensor, pad=(1, 1, 1, 1), mode='reflect')

    feat1 = torch.abs(img_tensor - img_padded[:, :, 0:-2, 0:-2])
    feat2 = torch.abs(img_tensor - img_padded[:, :, 0:-2, 1:-1])
    feat3 = torch.abs(img_tensor - img_padded[:, :, 1:-1, 0:-2])
    feat4 = torch.abs(img_tensor - img_padded[:, :, 2:, 2:])

    layer_indices=(feat4+feat3+feat1+feat2)/4
    if mode==0:
        return layer_indices
    else:
        layer_indices = (layer_indices == 0).to(torch.int64)
        return layer_indices

def gradient_homo_loss(output,target):
    mask=pred_compute_layer_indices(target,mode=1)  #这里提取出来均值区域的mask了
    output1=pred_compute_layer_indices(output,mode=0)
    output2=output1*mask
    zero=torch.zeros_like(output2,dtype=torch.float)
    loss1 = F.l1_loss(output1, zero)
    return loss1
