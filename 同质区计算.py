import torch
from skimage import io, filters

import torchvision.transforms.functional as TF

image_path_new = r"D:\code\QXSLAB_SAROPT\QXSLAB_SAROPT\train\NLopt\243.png"
image_new = io.imread(image_path_new)
image_path_new1 = r"D:\code\QXSLAB_SAROPT\QXSLAB_SAROPT\train\NLopt\241.png"
image_new1 = io.imread(image_path_new1)
image_path_new2 = r"D:\code\QXSLAB_SAROPT\QXSLAB_SAROPT\train\NLopt\243.png"
image_new2 = io.imread(image_path_new2)

# 首先，将每个NumPy数组图像转换为tensor
image_tensor = TF.to_tensor(image_new).unsqueeze(0)  # 增加一个batch维度
image_tensor1 = TF.to_tensor(image_new1).unsqueeze(0)  # 增加一个batch维度
image_tensor2 = TF.to_tensor(image_new2).unsqueeze(0)  # 增加一个batch维度


# 然后，将所有图像tensor堆叠成一个四维tensor
images_batch = torch.cat((image_tensor, image_tensor1, image_tensor2), dim=0)

image_tensors_normalized = images_batch / torch.max(images_batch)
# 初始化一个与输入形状相同的 tensor 来存储处理后的结果
uniform_regions_batch = torch.zeros_like(image_tensors_normalized)

# 对批次中的每个图像分别处理
for i in range(uniform_regions_batch.shape[0]):
    # 提取单个图像，并转换为 numpy 数组用于计算 Otsu 阈值
    single_image_normalized = image_tensors_normalized[i].numpy()
    threshold = filters.threshold_otsu(single_image_normalized)

    # 根据阈值处理图像，并保存结果
    if threshold > 0.45:
        uniform_regions_batch[i] = (image_tensors_normalized[i] > threshold * (1.05)).float()
    elif threshold < 0.45:
        # 更加严格的阈值
        uniform_regions_batch[i] = (image_tensors_normalized[i] < (threshold * 0.85)).float()

import matplotlib.pyplot as plt

# uniform_regions_batch 中每个图像的可视化
batch_size = uniform_regions_batch.shape[0]  # 获取批次大小

for i in range(batch_size):
    # 提取处理后的单个图像
    processed_image = uniform_regions_batch[i].numpy().squeeze()  # 移除批次维度，转换为NumPy数组，适用于灰度图像

    # 创建一个新的图形窗口
    plt.figure(figsize=(6, 6))

    # 显示图像
    plt.imshow(processed_image, cmap='gray')  # 使用灰度色彩映射
    plt.title(f'Processed Image {i+1}')
    plt.axis('off')  # 不显示坐标轴

    # 展示图像
    plt.show()

