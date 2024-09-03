import cv2
import numpy as np
import ctypes
import os

# 加载共享库
lib = ctypes.CDLL('./nearestPixel.so')

# 定义函数指针类型和返回值类型
c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
c_float_p = ctypes.POINTER(ctypes.c_float)
c_void_p = ctypes.c_void_p

lib.nparray_to_mat.argtypes = [c_float_p, ctypes.c_int, ctypes.c_int]
lib.nparray_to_mat.restype = c_void_p

lib.findClosestPixel.argtypes = [c_void_p, c_void_p, c_void_p, ctypes.c_int, ctypes.c_int]
lib.findClosestPixel.restype = None

lib.mat_to_nparray.argtypes = [c_void_p, c_float_p]
lib.mat_to_nparray.restype = None

lib.release_mat.argtypes = [ctypes.c_void_p]
lib.release_mat.restype = None

# 输入和输出文件夹路径
input_img_folder = r'../dataset/train/s2'
input_sar_folder = r'../dataset/train/s1'
output_folder = r'../dataset/train/NLsar'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的图像文件
for img_filename in os.listdir(input_img_folder):
    if img_filename.endswith('.png'):
        img_path = os.path.join(input_img_folder, img_filename)
        sar_path = os.path.join(input_sar_folder, img_filename)
       
        
        if os.path.exists(sar_path):
            # 载入图像
            img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            sar_np = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img_np = img_np.astype(np.uint8)
            img_np = cv2.fastNlMeansDenoising(img_np, None, 10, 5, 21)
            img_np = img_np.astype(np.float32)
            img_np = np.ascontiguousarray(img_np, dtype=np.float32)
            sar_np = np.ascontiguousarray(sar_np, dtype=np.float32)
            height, width = img_np.shape
            
            # 创建目标图像数组
            dst_np = np.zeros(img_np.shape, dtype=np.float32)
            
            # numpy.array 转换为 cv::Mat
            img_mat = lib.nparray_to_mat(img_np.ctypes.data_as(c_float_p), height, width)
            sar_mat = lib.nparray_to_mat(sar_np.ctypes.data_as(c_float_p), height, width)
            dst_mat = lib.nparray_to_mat(dst_np.ctypes.data_as(c_float_p), height, width)
            
            # 对 cv::Mat 进行处理
            lib.findClosestPixel(img_mat, sar_mat, dst_mat, 3, 21)
            
            # 将 cv::Mat 转换回 numpy.array
            lib.mat_to_nparray(dst_mat, dst_np.ctypes.data_as(c_float_p))
            lib.release_mat(img_mat)
            lib.release_mat(sar_mat)
            lib.release_mat(dst_mat)
            
            # 保存处理后的图像
            output_path = os.path.join(output_folder, img_filename)
            cv2.imwrite(output_path, dst_np.astype(np.uint8))

print("批量处理完成")
