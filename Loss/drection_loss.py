import torch
import torch.nn.functional as F
from torch import nn


def dice_loss(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def pred_compute_layer_indices(img_tensor):
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

    # 找出所有方向梯度都相同的位置

    max_values, _ = stacked_features.max(dim=1, keepdim=True)
    min_values, _ = stacked_features.min(dim=1, keepdim=True)
    no_gradient = (max_values == min_values)

    # 创建一个新的层，用1填充没有梯度的位置
    no_gradient_layer = no_gradient.to(torch.int64)

    # 对于stacked_features中的每个特征图，将no_gradient为True的位置设置为0
    #for i in range(stacked_features.shape[1]):
    #    stacked_features[:, i, :, :][no_gradient.squeeze(1)] = 0
        # 将新层添加到stacked_features中
    stacked_features_with_no_gradient = torch.cat([no_gradient_layer, stacked_features], dim=1)

    #layer_indices=torch.softmax(stacked_features_with_no_gradient,dim=1)

    return stacked_features



def target_compute_layer_indices(img_tensor):
    # 假设 img_tensor 是一个四维张量，大小为 batch_size x channels x height x width

    # 进行边缘填充，边缘宽度为1
    img_padded = F.pad(img_tensor, pad=(1, 1, 1, 1), mode='constant', value=0)

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

    # 找出所有方向梯度都相同的位置
    max_values, _ = stacked_features.max(dim=1, keepdim=True)
    min_values, _ = stacked_features.min(dim=1, keepdim=True)
    no_gradient = (max_values == min_values)

    # 创建一个新的层，用1填充没有梯度的位置
    no_gradient_layer = no_gradient.to(torch.int64)

    # 对于stacked_features中的每个特征图，将no_gradient为True的位置设置为0
    #for i in range(stacked_features.shape[1]):
    #    stacked_features[:, i, :, :][no_gradient.squeeze(1)] = 0
        # 将新层添加到stacked_features中
    stacked_features_with_no_gradient = torch.cat([no_gradient_layer, stacked_features], dim=1)

    # 找到每个像素最大值所在的层数
    layer_indices = torch.argmax(stacked_features , dim=1)

    return layer_indices
def gradient_direction_loss(pred, target):
    # Assuming pred and target are 4D tensors of size batch_size x channels x height x width
    #batch_size, _, height, width = pred.shape

    # Calculate gradient direction index for both prediction and target
    pred_index = pred_compute_layer_indices(pred)

    target_index = target_compute_layer_indices(target)

    # 创建交叉熵损失函数的实例
    criterion = nn.CrossEntropyLoss()
    loss=criterion(pred_index,target_index)
    print(loss)
    return loss
