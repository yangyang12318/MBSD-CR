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

"""
def visualize_batch(batch_tensor):
    for i in range(batch_tensor.shape[0]):
        # 处理当前批次
        x = pred_compute_layer_indices(batch_tensor[i:i + 1, :, :, :])
        x = np.asarray(x, dtype=np.uint8).squeeze()  # Squeeze是为了去掉不必要的维度

        # 使用 Plotly 的 imshow 函数来显示图像
        fig = px.imshow(x, color_continuous_scale='Viridis',
                        labels=dict(color="Value"),
                        title=f"Visualization of the Data with Plotly - Batch {i + 1}")
        fig.update_xaxes(title="Column")
        fig.update_yaxes(title="Row")

        # 显示图像
        fig.show()




# 读取图像
img_path = r"E:\doctor\SAR\code\data\QXSLAB_SAROPT\QXSLAB_SAROPT\\train\\NLsar\860.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found or path is incorrect")

# 增加一个维度作为channel，并重复图像四次
img = np.expand_dims(img, axis=0)  # Now it's (1, 256, 256)
img_repeated = np.repeat(img[np.newaxis, :, :, :], 4, axis=0)  # Now it's (4, 1, 256, 256)

# 将NumPy数组转换为PyTorch张量
img_tensor = torch.tensor(img_repeated, dtype=torch.float32)
print(img_tensor.shape)  # 应该打印出 (4, 1, 256, 256)

# 调用函数进行可视化
visualize_batch(img_tensor)


img_path = r"E:\doctor\SAR\code\data\QXSLAB_SAROPT\QXSLAB_SAROPT\\train\\NLsar\860.png"

# 使用cv2.IMREAD_GRAYSCALE来读取图像为灰度图
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 检查是否正确读取图像
if img is None:
    raise ValueError("Image not found or path is incorrect")

# 增加两个维度：batch_size和channels
img = img[np.newaxis, np.newaxis, :, :]

# 将NumPy数组转换为PyTorch张量，并转换数据类型
img_tensor = torch.tensor(img, dtype=torch.float32)

print(img_tensor.shape)
x = pred_compute_layer_indices(img_tensor)
x=np.asarray(x,dtype=np.uint8).squeeze()
print(x)
# 使用 Plotly 的 imshow 函数来显示图像
fig = px.imshow(x, color_continuous_scale='Viridis',
                labels=dict(color="Value"),
                title="Visualization of the Data with Plotly")
fig.update_xaxes(title="Column")
fig.update_yaxes(title="Row")

# 显示图像
fig.show()
"""
