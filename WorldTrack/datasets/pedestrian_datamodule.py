# ==================== 行人数据模块 ====================
# 本文件定义 PyTorch Lightning 的数据模块
# 负责管理训练、验证、测试数据集的创建和数据加载器配置

import os
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

# 导入具体的数据集类
from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from datasets.pedestrian_dataset import PedestrianDataset
from datasets.sampler import TemporalSampler


class PedestrianDataModule(pl.LightningDataModule):
    """
    行人检测数据模块
    
    功能：
    - 根据数据集路径自动识别数据集类型（Wildtrack/MultiviewX）
    - 创建训练、验证、测试数据集
    - 配置数据加载器（DataLoader）
    - 支持时序采样策略
    """
    
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",  # 数据集根目录
            batch_size: int = 2,  # 批次大小
            num_workers: int = 4,  # 数据加载的进程数
            resolution=None,  # BEV网格分辨率 (Y, Z, X)
            bounds=None,  # 世界坐标边界
            accumulate_grad_batches=8,  # 梯度累积批次数
    ):
        """
        初始化数据模块
        
        参数:
            data_dir: 数据集根目录路径
            batch_size: 每个批次的样本数
            num_workers: 数据加载的并行进程数
            resolution: BEV空间的网格分辨率
            bounds: 世界坐标系的物理边界
            accumulate_grad_batches: 梯度累积的批次数（影响有效batch size）
        """
        super().__init__()
        
        # 保存配置参数
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        
        # 从路径中提取数据集名称（用于自动识别数据集类型）
        self.dataset = os.path.basename(self.data_dir)

        # 初始化数据集变量
        self.data_predict = None  # 预测数据集
        self.data_val = None      # 验证数据集
        self.data_train = None    # 训练数据集

    def setup(self, stage: Optional[str] = None):
        """
        根据阶段设置相应的数据集
        
        参数:
            stage: 训练阶段标识
                - 'fit': 训练阶段，需要训练和验证数据集
                - 'validate': 验证阶段，只需要验证数据集
                - 'test': 测试阶段，需要测试数据集
                - 'predict': 预测阶段，需要预测数据集
        """
        # ==================== 步骤1：根据数据集路径自动识别数据集类型 ====================
        if 'wildtrack' in self.dataset.lower():
            # Wildtrack 数据集：7相机，480x1440网格，2000帧
            base = Wildtrack(self.data_dir)
        elif 'multiviewx' in self.dataset.lower():
            # MultiviewX 数据集：6相机，640x1000网格，400帧
            base = MultiviewX(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        # ==================== 步骤2：根据训练阶段创建相应的数据集 ====================
        if stage == 'fit':
            # 训练阶段：创建训练数据集
            self.data_train = PedestrianDataset(
                base,
                is_train=True,  # 启用训练模式（数据增强、90%数据）
                resolution=self.resolution,
                bounds=self.bounds,
            )
            
        if stage == 'fit' or stage == 'validate':
            # 训练或验证阶段：创建验证数据集
            self.data_val = PedestrianDataset(
                base,
                is_train=False,  # 验证模式（无数据增强、10%数据）
                resolution=self.resolution,
                bounds=self.bounds,
            )
            
        if stage == 'test':
            # 测试阶段：创建测试数据集
            self.data_test = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds
            )
            
        if stage == 'predict':
            # 预测阶段：创建预测数据集
            self.data_predict = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )

    def train_dataloader(self):
        """
        创建训练数据加载器
        
        特点:
        - 使用时序采样器（TemporalSampler）确保批次内的样本时序连续
        - 支持梯度累积
        - 启用内存固定（pin_memory）加速GPU传输
        
        返回:
            DataLoader: 训练数据加载器
        """
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # 时序采样器：确保同一批次内的样本在时间上连续
            # 这对于使用时序信息的模型很重要
            sampler=TemporalSampler(self.data_train, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,  # 固定内存，加速数据传输到GPU
        )

    def val_dataloader(self):
        """
        创建验证数据加载器
        
        配置与训练类似，但用于验证数据集
        
        返回:
            DataLoader: 验证数据加载器
        """
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_val, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        创建测试数据加载器
        
        注意：这里返回空，可能是因为测试逻辑在其他地方实现
        或者使用不同的测试流程
        
        返回:
            None: 暂未实现
        """
        return None  # 测试数据加载器暂未实现