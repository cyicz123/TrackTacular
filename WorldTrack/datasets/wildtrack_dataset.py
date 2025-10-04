# ==================== Wildtrack 数据集基类 ====================
# 本文件定义 Wildtrack 数据集的基础信息和标定数据加载
# Wildtrack 是一个多视角行人检测数据集，包含7个相机视角

import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset

# Wildtrack 数据集的相机内参文件名列表（7个相机）
intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
# Wildtrack 数据集的相机外参文件名列表（7个相机）
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']


class Wildtrack(VisionDataset):
    """
    Wildtrack 数据集基类
    
    数据集特点：
    - 7个相机视角，覆盖约12x36米的区域
    - 图像分辨率：1080x1920 (H x W)
    - 世界坐标网格：480x1440 (对应实际空间12m x 36m)
    - 帧数：2000帧（每5帧采样一次，实际400个独立帧）
    - 标注：行人位置使用 positionID（网格索引）
    
    坐标系说明：
    - 图像坐标：(H, W) = (1080, 1920)，按像素索引
    - 世界网格坐标：(i, j) = (480, 1440)，i对应高度(h)，j对应宽度(w)
    - 世界真实坐标：单位为厘米(cm)，通过变换矩阵从网格坐标转换
    """
    def __init__(self, root):
        """
        初始化 Wildtrack 数据集
        
        参数:
            root: 数据集根目录路径
        """
        super().__init__(root)
        # 图像形状：高度x宽度 = 1080x1920像素；C通道在前，H行在前，W列在前
        # 世界网格形状：N_row x N_col = 480 x 1440 (对应12m x 36m的区域)
        self.__name__ = 'Wildtrack'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 1440]  # H,W; N_row,N_col
        
        # 相机数量：7个，帧数：2000（实际每5帧采样，共400个关键帧）
        self.num_cam, self.num_frame = 7, 2000
        # 帧采样步长：每5帧取一帧
        self.frame_step = 5
        
        # 世界坐标系转换矩阵：从网格坐标到真实世界坐标（单位：厘米）
        # 变换公式：[x_world, y_world, 1]^T = worldcoord_from_worldgrid_mat @ [i, j, 1]^T
        # 矩阵解释：
        #   [0, 2.5, -300]  -> x_world = 2.5 * j - 300
        #   [2.5, 0, -900]  -> y_world = 2.5 * i - 900
        #   [0, 0, 1]       -> 齐次坐标
        # 每个网格单元 = 2.5cm x 2.5cm
        self.worldcoord_from_worldgrid_mat = np.array([[0, 2.5, -300], [2.5, 0, -900], [0, 0, 1]])
        
        # 加载所有相机的内参和外参矩阵
        # 返回两个元组：(内参矩阵列表, 外参矩阵列表)
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        """
        获取指定帧范围内所有相机的图像文件路径
        
        参数:
            frame_range: 帧索引范围（如 range(0, 400)）
        
        返回:
            img_fpaths: 嵌套字典 {相机ID: {帧ID: 图像路径}}
        """
        # 初始化存储结构：每个相机对应一个字典
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        
        # 遍历所有相机文件夹（Image_subsets目录下）
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            # 从文件夹名提取相机ID（如 "C1" -> cam=0）
            cam = int(camera_folder[-1]) - 1
            # 跳过超出范围的相机
            if cam >= self.num_cam:
                continue
            
            # 遍历该相机文件夹下的所有图像文件
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                # 从文件名提取帧号（如 "00000005.png" -> frame=5）
                frame = int(fname.split('.')[0])
                # 只保存在指定范围内的帧
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        """
        从 positionID 转换为世界网格坐标 (grid_x, grid_y)
        
        Wildtrack 使用一维的 positionID 来标识行人位置
        转换规则：pos = grid_x * 480 + grid_y
        
        参数:
            pos: positionID（一维索引，范围 0 ~ 691199）
        
        返回:
            [grid_x, grid_y]: 世界网格坐标，grid_x ∈ [0, 1440), grid_y ∈ [0, 480)
        """
        # positionID 到网格坐标的转换
        grid_y = pos % 480  # 行索引（高度方向）
        grid_x = pos // 480  # 列索引（宽度方向）
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        """
        加载指定相机的内参和外参矩阵
        
        参数:
            camera_i: 相机索引 (0~6)
        
        返回:
            intrinsic_matrix: 相机内参矩阵 (3x3)，包含焦距和主点
            extrinsic_matrix: 相机外参矩阵 (3x4)，包含旋转和平移
        """
        # ==================== 加载相机内参 ====================
        # 内参文件路径（使用零畸变的标定结果）
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        # 使用 OpenCV 读取 XML 格式的内参文件
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        # 提取内参矩阵 (3x3)
        # 格式：[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        # fx, fy: 焦距; cx, cy: 主点坐标
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        # ==================== 加载相机外参 ====================
        # 外参文件路径（XML格式）
        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        # 提取旋转向量 rvec（3x1，罗德里格斯格式）
        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        # 提取平移向量 tvec（3x1，单位：厘米）
        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        # 将旋转向量转换为旋转矩阵（3x3）
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        # 将平移向量重塑为列向量（3x1）
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        # 组合旋转和平移为外参矩阵 (3x4) = [R | t]
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix
