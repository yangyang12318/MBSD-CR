import torch
import torch.nn.functional as F

import numpy as np

def pred_compute_layer_indices(img_tensor,mode):
    # 假设 img_tensor 是一个四维张量，大小为 batch_size x channels x height x width

    # 进行边缘填充，边缘宽度为1
    img_padded = F.pad(img_tensor, pad=(1, 1, 1, 1), mode='reflect')

    # 提取所有方向的特征
    feat1 = torch.abs(img_tensor - img_padded[:, :, 0:-2, 0:-2])
    feat2 = torch.abs(img_tensor - img_padded[:, :, 0:-2, 1:-1])
    feat3 = torch.abs(img_tensor - img_padded[:, :, 0:-2, 2:])
    feat4 = torch.abs(img_tensor - img_padded[:, :, 1:-1, 0:-2])
    feat5 = torch.abs(img_tensor - img_padded[:, :, 1:-1, 2:])
    feat6 = torch.abs(img_tensor - img_padded[:, :, 2:, 0:-2])
    feat7 = torch.abs(img_tensor - img_padded[:, :, 2:, 1:-1])
    feat8 = torch.abs(img_tensor - img_padded[:, :, 2:, 2:])

    # 将所有特征图堆叠成一个新的4维数组
    stacked_features = torch.cat([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8], dim=1)

    # 将小于10的像素值设置为0
    stacked_features[stacked_features < 3] = 0
    # 找出所有方向梯度都相同的位置
    max_values, _ = stacked_features.max(dim=1, keepdim=True)
    min_values, _ = stacked_features.min(dim=1, keepdim=True)
    no_gradient = (max_values==min_values)  #(1,1,256,256)

    # 将新层添加到stacked_features中
    stacked_features_with_no_gradient = torch.cat([no_gradient, stacked_features], dim=1)

    layer_indices = torch.argmax(stacked_features_with_no_gradient, dim=1)

    if mode==0:
        return layer_indices
    # 把 layer_indices 中等于0的像素标注为1，其他像素为0
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
