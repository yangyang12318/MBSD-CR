
import math
import torch
import cv2
"""
def phase(InputImage, NumberScales, NumberAngles):
    '''
    计算图像相位 - 使用 PyTorch 在 GPU 上运行
    '''
    minWaveLength = 3
    mult = 1.6
    sigmaOnf = 0.75
    nrows, ncols = InputImage.shape


    # 复数的 Fourier 变换
    f_cv = torch.fft.fft2(InputImage)

    # 初始化输出矩阵
    EO = torch.zeros((nrows, ncols, NumberScales, NumberAngles), dtype=torch.cfloat)

    # 创建网格
    y = torch.arange(nrows).view(-1, 1).repeat(1, ncols)
    x = torch.arange(ncols).view(1, -1).repeat(nrows, 1)
    cy = nrows // 2
    cx = ncols // 2
    y = (y - cy) / nrows
    x = (x - cx) / ncols
    radius = torch.sqrt(x ** 2 + y ** 2)
    radius[cy, cx] = 1
    theta = torch.atan2(-y, x)
    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    # 创建滤波器
    annularBandpassFilters = torch.empty((nrows, ncols, NumberScales))
    filterorder = 15
    cutoff = .45
    normradius = radius / (abs(x).max() * 2)
    lowpassbutterworth = 1.0 / (1.0 + (normradius / cutoff) ** (2 * filterorder))

    for s in range(NumberScales):
        wavelength = minWaveLength * mult ** s
        fo = 1.0 / wavelength
        logGabor = torch.exp((-(torch.log(radius / fo)) ** 2) / (2 * (math.log(sigmaOnf) ** 2)))
        annularBandpassFilters[:, :, s] = logGabor * lowpassbutterworth
        annularBandpassFilters[cy, cx, s] = 0

    for o in range(NumberAngles):
        angl = torch.tensor(o * math.pi / NumberAngles)
        ds = sintheta * torch.cos(angl) - costheta * torch.sin(angl)
        dc = costheta * torch.cos(angl) + sintheta * torch.sin(angl)
        dtheta = torch.abs(torch.atan2(ds, dc))
        dtheta = torch.minimum(dtheta * NumberAngles / 2, torch.tensor(math.pi))
        spread = (torch.cos(dtheta) + 1) / 2

        for s in range(NumberScales):
            filter = annularBandpassFilters[:, :, s] * spread
            criticalfiltershift = torch.fft.fftshift(filter)
            MatrixEO = torch.fft.ifft2(criticalfiltershift * f_cv)

            # 提取 MatrixEO 的实部和虚部，确保它们是二维的
            MatrixEO_real = MatrixEO.real.squeeze(-1)
            MatrixEO_imag = MatrixEO.imag.squeeze(-1)

            # 现在可以安全地赋值
            EO[:, :, s, o] = MatrixEO_real + 1j * MatrixEO_imag

            yim, xim = InputImage.shape
            # 初始化卷积序列
            CS = torch.zeros((yim, xim, 6))
            # 计算卷积序列
            for j in range(6):
                for i in range(4):
                    CS[:, :, j] += torch.abs(EO[:, :, i, j])

            # 计算最大索引映射
            MIM = torch.argmax(CS, dim=2)

    return MIM





if __name__ == "__main__":


    # 读取图像并转换为 PyTorch 张量
    path = r"D:\code\QXSLAB_SAROPT\QXSLAB_SAROPT\val\opt\10.png"
    img = cv2.imread(path, 0)
    img_tensor = torch.from_numpy(img).float()
    print(img_tensor.shape)
    # 计算相位
    MIM = phase(img_tensor, 4, 6)



    # 归一化 MIM 到 0-1 范围
    MIM_min = MIM.min()
    MIM_max = MIM.max()
    MIM_normalized = (MIM - MIM_min) / (MIM_max - MIM_min)

    # 缩放到 0-255 范围并转换为整数类型以便显示
    MIM_scaled = (MIM_normalized * 255).byte()

    # 将处理后的图像移回 CPU 并转换为 NumPy 数组用于显示
    MIM_scaled_np = MIM_scaled.cpu().numpy()

    # 显示归一化后的图像
    cv2.imshow('Normalized MIM', MIM_scaled_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

import torch
import math

def phase(InputImage, NumberScales, NumberAngles):
    '''
    计算图像相位 - 使用 PyTorch 在 GPU 上运行，适用于四维的 InputImage
    '''
    minWaveLength = 3
    mult = 1.6
    sigmaOnf = 0.75
    batch_size, channels, nrows, ncols = InputImage.shape

    # 初始化输出矩阵
    MIM_batch = torch.zeros((batch_size, nrows, ncols), device=InputImage.device)

    for b in range(batch_size):
        # 提取当前图像
        current_image = InputImage[b, 0, :, :]  # 假设我们处理第一个通道

        # 复数的 Fourier 变换
        f_cv = torch.fft.fft2(current_image)

        # 初始化输出矩阵
        EO = torch.zeros((nrows, ncols, NumberScales, NumberAngles), dtype=torch.cfloat, device=InputImage.device)

        # 创建网格
        y = torch.arange(nrows, device=InputImage.device).view(-1, 1).repeat(1, ncols)
        x = torch.arange(ncols, device=InputImage.device).view(1, -1).repeat(nrows, 1)
        cy = nrows // 2
        cx = ncols // 2
        y = (y - cy) / nrows
        x = (x - cx) / ncols
        radius = torch.sqrt(x ** 2 + y ** 2)
        radius[cy, cx] = 1
        theta = torch.atan2(-y, x)
        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)

        # 创建滤波器
        annularBandpassFilters = torch.empty((nrows, ncols, NumberScales), device=InputImage.device)
        filterorder = 15
        cutoff = .45
        normradius = radius / (abs(x).max() * 2)
        lowpassbutterworth = 1.0 / (1.0 + (normradius / cutoff) ** (2 * filterorder))

        for s in range(NumberScales):
            wavelength = minWaveLength * mult ** s
            fo = 1.0 / wavelength
            logGabor = torch.exp((-(torch.log(radius / fo)) ** 2) / (2 * (math.log(sigmaOnf) ** 2)))
            annularBandpassFilters[:, :, s] = logGabor * lowpassbutterworth
            annularBandpassFilters[cy, cx, s] = 0

        for o in range(NumberAngles):
            angl = o * math.pi / NumberAngles
            ds = sintheta * torch.cos(angl) - costheta * torch.sin(angl)
            dc = costheta * torch.cos(angl) + sintheta * torch.sin(angl)
            dtheta = torch.abs(torch.atan2(ds, dc))
            dtheta = torch.minimum(dtheta * NumberAngles / 2, math.pi)
            spread = (torch.cos(dtheta) + 1) / 2

            for s in range(NumberScales):
                filter = annularBandpassFilters[:, :, s] * spread
                criticalfiltershift = torch.fft.fftshift(filter)
                MatrixEO = torch.fft.ifft2(criticalfiltershift * f_cv)

                # 提取 MatrixEO 的实部和虚部
                MatrixEO_real = MatrixEO.real
                MatrixEO_imag = MatrixEO.imag

                # 现在可以安全地赋值
                EO[:, :, s, o] = MatrixEO_real + 1j * MatrixEO_imag

        # 初始化卷积序列
        CS = torch.zeros((nrows, ncols, 6), device=InputImage.device)
        # 计算卷积序列
        for j in range(6):
            for i in range(4):
                CS[:, :, j] += torch.abs(EO[:, :, i, j])

        # 计算最大索引映射
        MIM = torch.argmax(CS, dim=2)

        # 将当前图像的结果存储到批处理输出中
        MIM_batch[b, :, :] = MIM

    return MIM_batch

