import torch
from skimage import  filters
import torch.nn.functional as F


def uniform_loss(opt_img,output,target):
    image_tensor=opt_img
    # Normalize the image tensor
    image_tensors_normalized = image_tensor / torch.max(image_tensor)
    # 初始化一个与输入形状相同的 tensor 来存储处理后的结果
    uniform_regions_batch = torch.zeros_like(image_tensors_normalized)

    # 对批次中的每个图像分别处理
    for i in range(uniform_regions_batch.shape[0]):
        # 提取单个图像，并转换为 numpy 数组用于计算 Otsu 阈值
        single_image_normalized = image_tensors_normalized[i].cpu().numpy()
        threshold = filters.threshold_otsu(single_image_normalized)

        # 根据阈值处理图像，并保存结果
        if threshold > 0.45:
            uniform_regions_batch[i] = (image_tensors_normalized[i] > threshold * (1.05)).float()
        elif threshold < 0.45:
            # 更加严格的阈值
            uniform_regions_batch[i] = (image_tensors_normalized[i] < (threshold * 0.85)).float()  #最后的uniform_regions_batch是（3，1，256，256）格式的

    pred=uniform_regions_batch*output
    target=uniform_regions_batch*target
    loss = F.l1_loss(pred,target)
    return loss