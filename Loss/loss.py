
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import regist_loss
eps = 1e-6

# ============================ #
#      Reconstruction loss     #
# ============================ #


@regist_loss
class L1():
    def __call__(self,  output_sar, data):
        mask = pred_compute_layer_indices(data['NLsar'])  # 这里提取出来均值区域的mask了
        # 将 mask 转换为布尔类型
        mask_bool = mask.to(torch.bool)
        target = torch.where(mask_bool, data['NLsar'], data['real_noisy'])
        sar_loss = F.l1_loss(output_sar, target)
        #sar_loss1 = F.l1_loss(output_sar, data['real_noisy'])
        return sar_loss  #+0.7*sar_loss1

@regist_loss
class L2():
    def __call__(self,  output_sar, data):

        sar_loss=F.mse_loss(output_sar, data['real_noisy'])
        loss=torch.sqrt(sar_loss)
        return loss


@regist_loss
class fft_loss(torch.nn.Module):
    def __call__(self, model_output, data):
        # 确保model_output和data在同一个设备上
        model_output = model_output
        clean_data = data['clean']

        # 执行傅里叶变换
        f_transform_output = torch.fft.fft2(model_output)
        f_transform_clean = torch.fft.fft2(clean_data)

        # 零频率分量移到中心
        f_shift_output = torch.fft.fftshift(f_transform_output)
        f_shift_clean = torch.fft.fftshift(f_transform_clean)

        # 计算幅度和相位
        amplitude_output = torch.abs(f_shift_output)
        amplitude_clean = torch.abs(f_shift_clean)
        phase_output = torch.angle(f_shift_output)
        phase_clean = torch.angle(f_shift_clean)

        # (b) 幅度和零相位的图像
        new_amplitude_image_output = torch.fft.ifft2(torch.fft.ifftshift(amplitude_output)).real
        new_amplitude_image_clean = torch.fft.ifft2(torch.fft.ifftshift(amplitude_clean)).real

        # (c) 相位和单位幅度的图像
        unit_amplitude = torch.ones_like(amplitude_output)
        new_phase_image_output = torch.fft.ifft2(torch.fft.ifftshift(unit_amplitude * torch.exp(1j * phase_output))).real
        new_phase_image_clean = torch.fft.ifft2(torch.fft.ifftshift(unit_amplitude * torch.exp(1j * phase_clean))).real

        # (d) 相位和平均幅度的图像
        mean_amplitude_output = amplitude_output.mean(dim=(-2, -1), keepdim=True)
        mean_amplitude_output = mean_amplitude_output.expand_as(amplitude_output)
        mean_amplitude_clean = amplitude_clean.mean(dim=(-2, -1), keepdim=True)
        mean_amplitude_clean = mean_amplitude_clean.expand_as(amplitude_clean)
        new_average_amplitude_image_output = torch.fft.ifft2(torch.fft.ifftshift(mean_amplitude_output * torch.exp(1j * phase_output))).real
        new_average_amplitude_image_clean = torch.fft.ifft2(torch.fft.ifftshift(mean_amplitude_clean * torch.exp(1j * phase_clean))).real

        # 计算三种差异的MSE损失
        loss_amplitude = F.mse_loss(new_amplitude_image_output, new_amplitude_image_clean)
        loss_phase = F.mse_loss(new_phase_image_output, new_phase_image_clean)
        loss_average_amplitude = F.mse_loss(new_average_amplitude_image_output, new_average_amplitude_image_clean)

        # 返回损失总和
        total_loss = loss_amplitude + loss_phase + loss_average_amplitude
        return total_loss


@regist_loss
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

    stacked_features[stacked_features < 3] = 0
    # 找出所有方向梯度都相同的位置
    max_values, _ = stacked_features.max(dim=1, keepdim=True)
    min_values, _ = stacked_features.min(dim=1, keepdim=True)
    no_gradient = (max_values==min_values)  #(1,1,256,256)



    # 将新层添加到stacked_features中
    stacked_features_with_no_gradient = torch.cat([no_gradient, stacked_features], dim=1)

    layer_indices = torch.argmax(stacked_features_with_no_gradient, dim=1)


    layer_indices = ((layer_indices == 0).to(torch.int64)).unsqueeze(1)

    return layer_indices

@regist_loss
class gradient_homo_loss():
    def __call__(self, output_sar, data):
        mask=pred_compute_layer_indices(data['NLsar'],mode=1)  #这里提取出来均值区域的mask了
        output1=pred_compute_layer_indices(output_sar,mode=0)
        output1=output1*mask
        zero=torch.zeros_like(output1,dtype=torch.float)
        loss1 = F.l1_loss(output1, zero)
        return loss1


