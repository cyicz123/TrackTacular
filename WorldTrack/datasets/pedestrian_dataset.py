# ==================== 行人数据集包装类 ====================
# 本文件将基础数据集（Wildtrack/MultiviewX）包装为PyTorch可用的数据集
# 主要功能：
# 1. 加载多视角图像和标注
# 2. 数据增强（resize、crop等）
# 3. 生成BEV和图像空间的ground truth
# 4. 坐标系转换（世界坐标 -> BEV网格坐标）

import os
import json
from operator import itemgetter

import torch
import numpy as np
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image

# 导入几何变换、基础工具和体素化工具
from utils import geom, basic, vox


class PedestrianDataset(VisionDataset):
    """
    行人检测数据集类
    
    功能说明：
    - 封装 Wildtrack/MultiviewX 等多视角行人数据集
    - 实现数据加载、预处理和ground truth生成
    - 支持训练时的数据增强
    """
    def __init__(
            self,
            base,  # 基础数据集对象（Wildtrack 或 MultiviewX）
            is_train=True,  # 是否为训练模式
            resolution=(160, 4, 250),  # BEV网格分辨率 (Y, Z, X)
            bounds=(-500, 500, -320, 320, 0, 2),  # 世界坐标边界 (xmin, xmax, ymin, ymax, zmin, zmax)
            final_dim: tuple = (720, 1280),  # 图像最终尺寸 (H, W)
            resize_lim: list = (0.8, 1.2),  # resize范围（数据增强用）
    ):
        """
        初始化行人数据集
        
        参数:
            base: 基础数据集对象（包含图像路径、标定等基础信息）
            is_train: 训练/验证模式标志
            resolution: BEV空间的体素网格分辨率
            bounds: BEV空间的实际物理边界（单位：厘米）
            final_dim: 输入图像的目标尺寸
            resize_lim: 数据增强时的resize范围
        """
        super().__init__(base.root)
        # 保存基础数据集的引用
        self.base = base
        # 提取基础信息：数据根目录、相机数量、总帧数
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        
        # 原始图像形状和世界网格形状（来自数据集标注）
        # MultiviewX: [1080, 1920], [640, 1000]
        # Wildtrack: [1080, 1920], [480, 1440]
        self.img_shape = base.img_shape
        self.worldgrid_shape = base.worldgrid_shape
        
        # 训练模式标志
        self.is_train = is_train
        # 世界坐标边界
        self.bounds = bounds
        # BEV网格分辨率
        self.resolution = resolution
        # 数据增强配置
        self.data_aug_conf = {'final_dim': final_dim, 'resize_lim': resize_lim}
        # 高斯核大小（用于生成热图）
        self.kernel_size = 1.5
        # 每帧最大目标数量
        self.max_objects = 60
        # 图像下采样倍数（编码器的stride）
        self.img_downsample = 4

        # 解包BEV网格尺寸
        self.Y, self.Z, self.X = self.resolution
        # 场景中心点（用于坐标变换的参考点）
        self.scene_centroid = torch.tensor((0., 0., 0.)).reshape([1, 3])

        # 初始化体素化工具（用于世界坐标<->BEV网格坐标的转换）
        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False)  # 不要求立方体网格

        # 划分训练/验证集：90% 训练，10% 验证
        if self.is_train:
            frame_range = range(0, int(self.num_frame * 0.9))
        else:
            frame_range = range(int(self.num_frame * 0.9), self.num_frame)

        # 获取所有图像文件路径
        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        # 初始化存储结构
        self.world_gt = {}  # 世界坐标系的ground truth
        self.imgs_gt = {}   # 图像空间的ground truth
        self.pid_dict = {}  # 行人ID映射字典
        # 加载标注数据
        self.download(frame_range)

        # ground truth文件路径
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        # 准备评估用的ground truth
        self.prepare_gt()

        # 相机标定数据
        self.calibration = {}
        # 初始化标定参数
        self.setup()

    def setup(self):
        """
        初始化相机标定参数
        将相机内参和外参转换为4x4齐次变换矩阵形式
        """
        # ==================== 处理相机内参 ====================
        # 将所有相机的内参矩阵堆叠：(num_cameras, 3, 3)
        intrinsic = torch.tensor(np.stack(self.base.intrinsic_matrices, axis=0), dtype=torch.float32)
        # 转换为4x4齐次形式：(num_cameras, 4, 4)
        # 内参矩阵格式：[[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic)).squeeze()
        self.calibration['intrinsic'] = intrinsic
        
        # ==================== 处理相机外参 ====================
        # 初始化为单位矩阵
        self.calibration['extrinsic'] = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
        # 填充旋转和平移部分（前3行）
        # 外参矩阵格式：[[R | t], [0, 0, 0, 1]]，R是3x3旋转矩阵，t是3x1平移向量
        self.calibration['extrinsic'][:, :3] = torch.tensor(
            np.stack(self.base.extrinsic_matrices, axis=0), dtype=torch.float32)

    def prepare_gt(self):
        """
        准备评估用的ground truth文件
        将所有帧的行人位置保存为txt格式：[frame_id, grid_x, grid_y]
        用于后续计算MODA、MODP等评估指标
        """
        og_gt = []
        # 遍历所有标注文件
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            # 加载该帧的所有行人标注（JSON格式）
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            
            # 处理每个行人
            for single_pedestrian in all_pedestrians:
                # 定义辅助函数：检查行人是否在某个相机视野内
                def is_in_cam(cam):
                    # bbox坐标为-1表示该相机看不到该行人
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                # 计算有多少个相机能看到该行人
                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                # 如果没有相机能看到，跳过
                if not in_cam_range:
                    continue
                
                # 从positionID转换为世界网格坐标
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                # 保存：[帧号, x坐标, y坐标]
                og_gt.append(np.array([frame, grid_x, grid_y]))
        
        # 将所有ground truth堆叠成数组
        og_gt = np.stack(og_gt, axis=0)
        # 确保目录存在
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        # 保存为文本文件（整数格式）
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        """
        加载指定帧范围内的所有标注数据
        将标注转换为世界坐标和图像空间的bbox
        
        参数:
            frame_range: 要加载的帧范围
        """
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        # 遍历所有标注文件
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            # 只处理指定范围内的帧
            if frame in frame_range:
                num_frame += 1
                # 加载该帧的标注
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                
                # 初始化该帧的数据存储
                world_pts, world_pids = [], []  # 世界坐标的点和ID
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]  # 每个相机的bbox和ID

                # 处理该帧的每个行人
                for pedestrian in all_pedestrians:
                    # 获取世界网格坐标
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    
                    # 为行人分配唯一ID（建立personID到连续整数的映射）
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                    
                    num_world_bbox += 1
                    # 保存世界坐标位置和ID
                    world_pts.append((grid_x, grid_y))
                    world_pids.append(pedestrian['personID'])
                    
                    # 遍历所有相机，保存该行人在各相机中的bbox
                    for cam in range(self.num_cam):
                        # 检查该行人是否在该相机视野内（bbox不为-1）
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            # 提取bbox坐标（左上角和右下角）
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pedestrian['personID'])
                            num_imgs_bbox += 1
                
                # 将该帧的世界坐标标注转换为tensor并保存
                self.world_gt[frame] = (torch.tensor(world_pts, dtype=torch.float32),
                                        torch.tensor(world_pids, dtype=torch.float32))
                
                # 将该帧各相机的图像标注转换为tensor并保存
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # bbox格式：(x1, y1, x2, y2) 左上角和右下角坐标
                    self.imgs_gt[frame][cam] = (torch.tensor(img_bboxs[cam]), torch.tensor(img_pids[cam]))

    def get_bev_gt(self, mem_pts, mem_pts_prev, pids, pids_pre):
        """
        生成BEV空间的ground truth
        
        参数:
            mem_pts: 当前帧的BEV网格坐标 (1, N, 3)
            mem_pts_prev: 前一帧的BEV网格坐标 (1, M, 3)
            pids: 当前帧的行人ID列表
            pids_pre: 前一帧的行人ID列表
        
        返回:
            center: 中心点热图 (1, Y, X)，使用高斯分布表示目标位置
            valid_mask: 有效位置掩码 (1, Y, X)
            person_ids: 每个位置的行人ID (1, Y, X)
            offset: 偏移量 (4, Y, X)
                    - offset[0:2]: 亚像素偏移（当前帧）
                    - offset[2:4]: 时序偏移（相对于前一帧）
        """
        # 初始化输出张量
        center = torch.zeros((1, self.Y, self.X), dtype=torch.float32)  # 中心点热图
        valid_mask = torch.zeros((1, self.Y, self.X), dtype=torch.bool)  # 有效掩码
        offset = torch.zeros((4, self.Y, self.X), dtype=torch.float32)  # 偏移量（前2维：空间偏移，后2维：时序偏移）
        person_ids = torch.zeros((1, self.Y, self.X), dtype=torch.long)  # 行人ID

        # 将前一帧的行人位置构建为字典：{pid: position}
        prev_pts = dict(zip(pids_pre.int().tolist(), mem_pts_prev[0]))

        # 遍历当前帧的每个行人
        for pts, pid in zip(mem_pts[0], pids):
            # 提取中心点坐标（x, y）
            ct = pts[:2]
            # 取整得到网格索引
            ct_int = ct.int()

            # 边界检查：跳过超出BEV网格范围的点
            if ct_int[0] < 0 or ct_int[0] >= self.X or ct_int[1] < 0 or ct_int[1] >= self.Y:
                continue

            # ==================== 生成中心点热图 ====================
            # 使用高斯核在中心点位置绘制热图（类似CenterNet）
            for c in center:
                basic.draw_umich_gaussian(c, ct_int, self.kernel_size)
            
            # ==================== 标记有效位置 ====================
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            
            # ==================== 计算亚像素偏移 ====================
            # 存储小数部分，用于回归精确位置
            # 例如：如果 ct = (10.3, 20.7)，ct_int = (10, 20)
            #      则 offset = (0.3, 0.7)
            offset[:2, ct_int[1], ct_int[0]] = ct - ct_int
            
            # ==================== 保存行人ID ====================
            person_ids[:, ct_int[1], ct_int[0]] = pid

            # ==================== 计算时序偏移（用于跟踪）====================
            # 如果该行人在前一帧也出现过
            if pid in pids_pre:
                # 计算从前一帧到当前帧的位移
                t_off = prev_pts[pid.int().item()][:2] - ct_int
                # 如果位移过大（>15个网格），认为不可靠，跳过
                if t_off.abs().max() > 15:
                    continue
                # 保存时序偏移
                offset[2:, ct_int[1], ct_int[0]] = t_off

        return center, valid_mask, person_ids, offset

    def get_img_gt(self, img_pts, img_pids, sx, sy, crop):
        """
        生成图像空间的ground truth
        
        参数:
            img_pts: 图像中的bbox坐标 (N, 4)，格式 [xmin, ymin, xmax, ymax]
            img_pids: 对应的行人ID
            sx, sy: 图像resize的缩放比例
            crop: 裁剪区域 (crop_w, crop_h, crop_w+fW, crop_h+fH)
        
        返回:
            center: 中心点热图 (3, H, W)，分别对应脚部、中心、头部
            offset: 亚像素偏移 (2, H, W)
            size: bbox尺寸 (2, H, W)，包含宽度和高度
            person_ids: 行人ID (1, H, W)
            valid_mask: 有效掩码 (1, H, W)
        """
        # 计算特征图尺寸（原图下采样后）
        H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
        W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)
        
        # 初始化输出张量
        center = torch.zeros((3, H, W), dtype=torch.float32)  # 3通道：脚、中心、头
        offset = torch.zeros((2, H, W), dtype=torch.float32)  # 亚像素偏移
        size = torch.zeros((2, H, W), dtype=torch.float32)    # bbox尺寸
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)  # 有效掩码
        person_ids = torch.zeros((1, H, W), dtype=torch.long)  # 行人ID

        # ==================== 将bbox坐标变换到特征图空间 ====================
        # 步骤：原始坐标 -> resize -> crop -> 下采样
        xmin = (img_pts[:, 0] * sx - crop[0]) / self.img_downsample
        ymin = (img_pts[:, 1] * sy - crop[1]) / self.img_downsample
        xmax = (img_pts[:, 2] * sx - crop[0]) / self.img_downsample
        ymax = (img_pts[:, 3] * sy - crop[1]) / self.img_downsample

        # 计算关键点位置
        center_pts = np.stack(((xmin + xmax) / 2, (ymin + ymax) / 2), axis=1)  # bbox中心
        center_pts = torch.tensor(center_pts, dtype=torch.float32)
        
        size_pts = np.stack(((-xmin + xmax), (-ymin + ymax)), axis=1)  # bbox尺寸（宽、高）
        size_pts = torch.tensor(size_pts, dtype=torch.float32)
        
        foot_pts = np.stack(((xmin + xmax) / 2, ymin), axis=1)  # 脚部位置（bbox底部中心）
        foot_pts = torch.tensor(foot_pts, dtype=torch.float32)
        
        head_pts = np.stack(((xmin + xmax) / 2, ymax), axis=1)  # 头部位置（bbox顶部中心）
        head_pts = torch.tensor(head_pts, dtype=torch.float32)

        # ==================== 为每个目标生成ground truth ====================
        for pt_idx, (pid, wh) in enumerate(zip(img_pids, size_pts)):
            # 在关键点位置绘制高斯热图（这里只使用脚部）
            for idx, pt in enumerate((foot_pts[pt_idx], )):  # 可扩展为 (foot, center, head)
                # 边界检查
                if pt[0] < 0 or pt[0] >= W or pt[1] < 0 or pt[1] >= H:
                    continue
                # 绘制高斯核
                basic.draw_umich_gaussian(center[idx], pt.int(), self.kernel_size)

            # 使用脚部位置作为主要回归点
            ct_int = foot_pts[pt_idx].int()
            # 边界检查
            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue
            
            # 标记有效位置
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            # 保存亚像素偏移
            offset[:, ct_int[1], ct_int[0]] = foot_pts[pt_idx] - ct_int
            # 保存bbox尺寸
            size[:, ct_int[1], ct_int[0]] = wh
            # 保存行人ID
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, offset, size, person_ids, valid_mask

    def sample_augmentation(self):
        """
        采样数据增强参数（resize和crop）
        
        训练模式：
        - 随机resize到目标尺寸的 0.8x ~ 1.2x
        - 随机裁剪到目标尺寸
        
        验证模式：
        - 直接resize到目标尺寸
        - 不进行裁剪
        
        返回:
            resize_dims: resize后的尺寸 (W, H)
            crop: 裁剪区域 (crop_w, crop_h, crop_w+fW, crop_h+fH)
        """
        # 获取目标尺寸
        fH, fW = self.data_aug_conf['final_dim']
        
        if self.is_train:
            # ==================== 训练模式：随机数据增强 ====================
            # 随机选择resize比例（0.8 ~ 1.2）
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims

            # 计算居中裁剪的起始位置
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            # 添加随机偏移（使裁剪位置不完全居中）
            crop_offset = int(self.data_aug_conf['resize_lim'][0] * self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            # 定义裁剪区域：(左上x, 左上y, 右下x, 右下y)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:
            # ==================== 验证模式：直接resize ====================
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        
        return resize_dims, crop

    def get_image_data(self, frame, cameras):
        """
        加载并处理指定帧的所有相机图像数据
        
        参数:
            frame: 帧索引
            cameras: 相机ID列表
        
        返回:
            imgs: 图像张量 (num_cameras, 3, H, W)
            intrins: 内参矩阵 (num_cameras, 4, 4)
            extrins: 外参矩阵 (num_cameras, 4, 4)
            centers: 中心点热图 (num_cameras, 3, H/8, W/8)
            offsets: 偏移量 (num_cameras, 2, H/8, W/8)
            sizes: bbox尺寸 (num_cameras, 2, H/8, W/8)
            pids: 行人ID (num_cameras, 1, H/8, W/8)
            valids: 有效掩码 (num_cameras, 1, H/8, W/8)
        """
        # 初始化存储列表
        imgs, intrins, extrins = [], [], []
        centers, offsets, sizes, pids, valids = [], [], [], [], []
        
        # 遍历每个相机
        for cam in cameras:
            # ==================== 步骤1：加载图像 ====================
            img = Image.open(self.img_fpaths[cam][frame]).convert('RGB')
            W, H = img.size

            # ==================== 步骤2：采样数据增强参数 ====================
            resize_dims, crop = self.sample_augmentation()
            # 计算缩放比例
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            # ==================== 步骤3：调整相机标定参数 ====================
            # 获取原始内参和外参
            extrin = self.calibration['extrinsic'][cam]
            intrin = self.calibration['intrinsic'][cam]
            
            # 根据resize缩放内参矩阵
            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            # 提取内参的各个分量
            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))

            # 根据crop调整主点位置
            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            # 重新组合内参矩阵
            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)  # 形状: (4, 4)
            
            # ==================== 步骤4：应用图像变换 ====================
            img = basic.img_transform(img, resize_dims, crop)

            # 保存图像和标定参数
            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

            # ==================== 步骤5：生成图像空间的ground truth ====================
            # 获取该相机该帧的标注
            img_pts, img_pids = self.imgs_gt[frame][cam]
            # 生成ground truth（应用相同的变换）
            center_img, offset_img, size_img, pid_img, valid_img = self.get_img_gt(img_pts, img_pids, sx, sy, crop)

            # 保存ground truth
            centers.append(center_img)
            offsets.append(offset_img)
            sizes.append(size_img)
            pids.append(pid_img)
            valids.append(valid_img)

        # 将列表堆叠为张量并返回
        return torch.stack(imgs), torch.stack(intrins), torch.stack(extrins), torch.stack(centers), torch.stack(
            offsets), torch.stack(sizes), torch.stack(pids), torch.stack(valids)

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.world_gt.keys())

    def __getitem__(self, index):
        """
        获取数据集中的一个样本（核心方法）
        
        这是整个数据加载流程的核心，完成以下任务：
        1. 加载多视角图像和相机标定
        2. 处理世界坐标系的标注
        3. 生成BEV和图像空间的ground truth
        4. 应用数据增强
        
        参数:
            index: 样本索引
        
        返回:
            item: 输入数据字典
            target: ground truth字典
        """
        # ==================== 步骤1：确定当前帧和前一帧 ====================
        frame = list(self.world_gt.keys())[index]  # 当前帧ID
        pre_frame = list(self.world_gt.keys())[max(index - 1, 0)]  # 前一帧ID（用于时序信息）
        cameras = list(range(self.num_cam))  # 所有相机ID列表

        # ==================== 步骤2：加载多视角图像数据 ====================
        imgs, intrins, extrins, centers_img, offsets_img, sizes_img, pids_img, valids_img \
            = self.get_image_data(frame, cameras)

        # ==================== 步骤3：构建世界坐标变换矩阵 ====================
        # 创建4x4齐次变换矩阵：从世界网格坐标到世界真实坐标
        worldcoord_from_worldgrid = torch.eye(4)
        # 获取2D变换矩阵（来自数据集定义）
        worldcoord_from_worldgrid2d = torch.tensor(self.base.worldcoord_from_worldgrid_mat, dtype=torch.float32)
        # 填充旋转和平移部分
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]  # 2x2旋转缩放
        worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]   # 2x1平移
        # 计算逆变换：从世界真实坐标到世界网格坐标
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)

        # ==================== 步骤4：获取世界坐标系的标注 ====================
        # 当前帧的标注：(N, 2) 网格坐标 + (N,) 行人ID
        worldgrid_pts_org, world_pids = self.world_gt[frame]
        # 前一帧的标注（用于计算时序偏移）
        worldgrid_pts_pre, world_pid_pre = self.world_gt[pre_frame]

        # 将2D网格坐标扩展为3D齐次坐标：(N, 2) -> (1, N, 3)
        # 添加z=0维度，因为行人都在地面上
        worldgrid_pts = torch.cat((worldgrid_pts_org, torch.zeros_like(worldgrid_pts_org[:, 0:1])), dim=1).unsqueeze(0)
        worldgrid_pts_pre = torch.cat((worldgrid_pts_pre, torch.zeros_like(worldgrid_pts_pre[:, 0:1])), dim=1)

        # ==================== 步骤5：训练时的数据增强（坐标扰动）====================
        if self.is_train:
            # 创建随机的刚体变换（只有平移，没有旋转）
            Rz = torch.eye(3)  # 单位旋转矩阵
            scene_center = torch.tensor([0., 0., 0.], dtype=torch.float32)
            off = 0.25  # 平移范围：±0.25个网格单元
            # 随机平移场景中心
            scene_center[:2].uniform_(-off, off)
            # 构建4x4变换矩阵
            augment = geom.merge_rt(Rz.unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()
            # 应用变换到坐标系变换矩阵
            worldgrid_T_worldcoord = torch.matmul(augment, worldgrid_T_worldcoord)
            # 应用变换到当前帧的点坐标
            worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0), worldgrid_pts)

        # ==================== 步骤6：坐标系转换（世界坐标 -> BEV网格坐标）====================
        # 将世界坐标转换为BEV网格的内存坐标
        # Ref2Mem: 参考坐标系 -> 内存坐标系（BEV网格索引）
        mem_pts = self.vox_util.Ref2Mem(worldgrid_pts, self.Y, self.Z, self.X)
        mem_pts_pre = self.vox_util.Ref2Mem(worldgrid_pts_pre.unsqueeze(0), self.Y, self.Z, self.X)
        
        # ==================== 步骤7：生成BEV空间的ground truth ====================
        center_bev, valid_bev, pid_bev, offset_bev = self.get_bev_gt(mem_pts, mem_pts_pre, world_pids, world_pid_pre)

        # ==================== 步骤8：准备评估用的网格ground truth ====================
        # 创建固定大小的数组，存储原始网格坐标（用于评估）
        grid_gt = torch.zeros((self.max_objects, 3), dtype=torch.long)
        # 填充实际的目标：[grid_x, grid_y, person_id]
        grid_gt[:worldgrid_pts.shape[1], :2] = worldgrid_pts_org
        grid_gt[:worldgrid_pts.shape[1], 2] = world_pids

        # ==================== 步骤9：构建最终的输入和目标数据 ====================
        # 输入数据字典
        item = {
            'img': imgs,  # 多视角图像 (S, 3, H, W)
            'intrinsic': intrins,  # 内参矩阵 (S, 4, 4) 
            'extrinsic': extrins,  # 外参矩阵 (S, 4, 4)
            'ref_T_global': worldgrid_T_worldcoord,  # 参考系到全局坐标的变换 (4, 4)
            'frame': frame // self.base.frame_step,  # 标准化的帧号
            'sequence_num': int(0),  # 序列号（单序列数据集为0）
            'grid_gt': grid_gt,  # 网格ground truth，用于评估 (max_objects, 3)
        }

        # ground truth字典
        target = {
            # ==================== BEV空间的标注 ====================
            'valid_bev': valid_bev,    # 有效位置掩码 (1, Y, X)
            'center_bev': center_bev,  # 中心点热图 (1, Y, X)
            'offset_bev': offset_bev,  # 偏移量（空间+时序） (4, Y, X)
            'pid_bev': pid_bev,        # 行人ID (1, Y, X)
            
            # ==================== 图像空间的标注 ====================
            'center_img': centers_img,  # 中心点热图 (S, 3, H/8, W/8)
            'offset_img': offsets_img,  # 偏移量 (S, 2, H/8, W/8)
            'size_img': sizes_img,      # bbox尺寸 (S, 2, H/8, W/8)
            'valid_img': valids_img,    # 有效掩码 (S, 1, H/8, W/8)
            'pid_img': pids_img         # 行人ID (S, 1, H/8, W/8)
        }

        return item, target
