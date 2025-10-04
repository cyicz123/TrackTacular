# ==================== TrackTacular 主训练模型 ====================
# 本文件定义了完整的多视角3D目标检测和跟踪训练流程
# 基于 PyTorch Lightning 框架，集成了：
# 1. 多种模型架构（Liftnet, MVDet, Segnet, BEVFormer）
# 2. 损失计算和优化
# 3. 训练、验证、测试流程
# 4. 评估指标计算（MODA, MODP, MOTA等）
# 5. 多目标跟踪

import os.path as osp
import torch
import lightning as pl
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import numpy as np

# 导入各种模型架构
from models import Segnet, MVDet, Liftnet, Bevformernet
# 导入损失函数
from models.loss import FocalLoss, compute_rot_loss
# 导入多目标跟踪器
from tracking.multitracker import JDETracker
# 导入工具函数
from utils import vox, basic, decode
# 导入评估工具
from evaluation.mod import modMetricsCalculator
from evaluation.mot_bev import mot_metrics


class WorldTrackModel(pl.LightningModule):
    """
    WorldTrack 主训练模型
    
    这是整个系统的核心类，继承自 PyTorch Lightning 的 LightningModule
    负责：
    - 模型初始化和前向传播
    - 损失计算和反向传播  
    - 训练/验证/测试流程管理
    - 评估指标计算
    - 多目标跟踪
    """
    def __init__(
            self,
            model_name='segnet',  # 模型架构名称
            encoder_name='res18',  # 编码器backbone名称
            learning_rate=0.001,  # 学习率
            resolution=(200, 4, 200),  # BEV网格分辨率 (Y, Z, X)
            bounds=(-75, 75, -75, 75, -1, 5),  # 世界坐标边界 (xmin, xmax, ymin, ymax, zmin, zmax)
            num_cameras=None,  # 相机数量
            depth=(100, 2.0, 25),  # 深度配置 (D, DMIN, DMAX)
            scene_centroid=(0.0, 0.0, 0.0),  # 场景中心点
            max_detections=60,  # 每帧最大检测数量
            conf_threshold=0.5,  # 置信度阈值
            num_classes=1,  # 目标类别数（行人检测为1）
            use_temporal_cache=True,  # 是否使用时序缓存
            z_sign=1,  # Z轴方向符号
            feat2d_dim=128,  # 2D特征维度
    ):
        """
        初始化 WorldTrack 模型
        
        参数说明:
            model_name: 选择的模型架构 ('segnet', 'liftnet', 'mvdet', 'bevformer')
            encoder_name: 图像编码器类型 ('res18', 'res50', 'res101', 'effb0', 'effb4', 'swin_t')
            learning_rate: 优化器学习率
            resolution: BEV空间的体素网格分辨率
            bounds: BEV空间的物理边界（单位：米或厘米，取决于数据集）
            num_cameras: 相机数量（None表示自动从数据获取）
            depth: 深度相关配置（离散化层数、最小深度、最大深度）
            scene_centroid: 场景的几何中心，用于坐标变换
            max_detections: 单帧最大目标数量限制
            conf_threshold: 检测置信度阈值
            num_classes: 目标类别数量
            use_temporal_cache: 是否缓存前一帧的BEV特征用于时序融合
            z_sign: Z轴方向（1表示向上为正，-1表示向下为正）
            feat2d_dim: 2D图像特征的维度
        """
        super().__init__()
        
        # ==================== 保存基础配置参数 ====================
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.Y, self.Z, self.X = self.resolution  # 解包BEV网格尺寸
        self.bounds = bounds
        self.max_detections = max_detections
        self.D, self.DMIN, self.DMAX = depth  # 解包深度配置
        self.conf_threshold = conf_threshold

        # ==================== 初始化损失函数 ====================
        # 使用 Focal Loss 处理中心点检测的类别不平衡问题
        self.center_loss_fn = FocalLoss()

        # ==================== 时序缓存配置 ====================
        # 用于存储前几帧的BEV特征，实现时序信息融合
        self.use_temporal_cache = use_temporal_cache
        self.max_cache = 32  # 最大缓存帧数
        # 缓存帧索引，-2表示未使用的缓存槽
        self.temporal_cache_frames = -2 * torch.ones(self.max_cache, dtype=torch.long)
        self.temporal_cache = None  # 实际的特征缓存

        # ==================== 测试和评估相关变量 ====================
        # 用于存储测试过程中的检测和跟踪结果，最后统一计算评估指标
        self.moda_gt_list, self.moda_pred_list = [], []  # MODA评估用的GT和预测
        self.mota_gt_list, self.mota_pred_list = [], []  # MOTA评估用的GT和预测  
        self.mota_seq_gt_list, self.mota_seq_pred_list = [], []  # 序列级MOTA评估
        self.frame = 0  # 当前处理的帧号
        # 初始化多目标跟踪器
        self.test_tracker = JDETracker(conf_thres=self.conf_threshold)

        # ==================== 模型架构初始化 ====================
        # 处理相机数量参数（0表示None）
        num_cameras = None if num_cameras == 0 else num_cameras
        
        # 根据指定的模型名称初始化对应的模型架构
        if model_name == 'segnet':
            # Segnet: 基于分割的多视角检测模型
            self.model = Segnet(self.Y, self.Z, self.X, num_cameras=num_cameras, feat2d_dim=feat2d_dim,
                                encoder_type=self.encoder_name, num_classes=num_classes, z_sign=z_sign)
        elif model_name == 'liftnet':
            # Liftnet: 基于深度提升的多视角检测模型（本项目主要使用）
            self.model = Liftnet(self.Y, self.Z, self.X, encoder_type=self.encoder_name, feat2d_dim=feat2d_dim,
                                 DMIN=self.DMIN, DMAX=self.DMAX, D=self.D, num_classes=num_classes, z_sign=z_sign,
                                 num_cameras=num_cameras)
        elif model_name == 'bevformer':
            # BEVFormer: 基于Transformer的BEV检测模型
            self.model = Bevformernet(self.Y, self.Z, self.X, feat2d_dim=feat2d_dim,
                                      encoder_type=self.encoder_name, num_classes=num_classes, z_sign=z_sign)
        elif model_name == 'mvdet':
            # MVDet: 经典的多视角检测模型
            self.model = MVDet(self.Y, self.Z, self.X, encoder_type=self.encoder_name,
                               num_cameras=num_cameras, num_classes=num_classes)
        else:
            raise ValueError(f'Unknown model name {self.model_name}')

        # ==================== 几何工具初始化 ====================
        # 场景中心点，用于坐标变换的参考点
        self.scene_centroid = torch.tensor(scene_centroid, device=self.device).reshape([1, 3])
        # 体素化工具：处理世界坐标与BEV网格坐标的转换
        self.vox_util = vox.VoxelUtil(self.Y, self.Z, self.X, scene_centroid=self.scene_centroid, bounds=self.bounds)
        
        # 保存所有超参数到checkpoint中
        self.save_hyperparameters()

    def forward(self, item):
        """
        模型前向传播
        
        参数:
            item: 输入数据字典，包含：
                - img: 多视角图像 (B, S, C, H, W)
                - intrinsic: 相机内参 (B, S, 4, 4)
                - extrinsic: 相机外参 (B, S, 4, 4)
                - ref_T_global: 参考系到全局坐标变换 (B, 4, 4)
                - frame: 帧号
        
        返回:
            output: 模型输出字典，包含检测结果
        """
        # ==================== 加载时序缓存 ====================
        # 尝试从缓存中加载前一帧的BEV特征
        prev_bev = self.load_cache(item['frame'].cpu())

        # ==================== 模型前向传播 ====================
        output = self.model(
            rgb_cams=item['img'],  # 多视角RGB图像
            pix_T_cams=item['intrinsic'],  # 像素到相机坐标的变换矩阵
            cams_T_global=item['extrinsic'],  # 相机到全局坐标的变换矩阵
            ref_T_global=item['ref_T_global'],  # 参考系到全局坐标的变换矩阵
            vox_util=self.vox_util,  # 体素化工具
            prev_bev=prev_bev,  # 前一帧的BEV特征（用于时序融合）
        )

        # ==================== 更新时序缓存 ====================
        if self.use_temporal_cache:
            # 将当前帧的BEV特征存入缓存供下一帧使用
            self.store_cache(item['frame'].cpu(), output['bev_raw'].clone().detach())

        return output

    def load_cache(self, frames):
        idx = []
        for frame in frames:
            i = (frame - 1 == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            if i.nelement() == 1:
                idx.append(i.item())
        if len(idx) != len(frames):
            return None
        else:
            return self.temporal_cache[idx]

    def store_cache(self, frames, bev_feat):
        if self.temporal_cache is None:
            shape = list(bev_feat.shape)
            shape[0] = self.max_cache
            self.temporal_cache = torch.zeros(shape, device=bev_feat.device, dtype=bev_feat.dtype)

        for frame, feat in zip(frames, bev_feat):
            i = (frame - 1 == self.temporal_cache_frames).nonzero(as_tuple=True)[0]
            # Choose unfilled cache slot
            if i.nelement() == 0:
                i = (self.temporal_cache_frames == -2).nonzero(as_tuple=True)[0]
            # Choose random cache slot
            if i.nelement() == 0:
                i = torch.randint(self.max_cache, (1, 1))

            self.temporal_cache[i[0]] = feat
            self.temporal_cache_frames[i[0]] = frame

    def loss(self, target, output):
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        center_img_e = output['img_center']

        valid_g = target['valid_bev']
        center_g = target['center_bev']
        offset_g = target['offset_bev']

        B, S = target['center_img'].shape[:2]
        center_img_g = basic.pack_seqdim(target['center_img'], B)

        center_loss = self.center_loss_fn(basic.sigmoid(center_e), center_g)
        offset_loss = torch.abs(offset_e[:, :2] - offset_g[:, :2]).sum(dim=1, keepdim=True)
        offset_loss = basic.reduce_masked_mean(offset_loss, valid_g)
        tracking_loss = torch.nn.functional.smooth_l1_loss(
            offset_e[:, 2:], offset_g[:, 2:], reduction='none').sum(dim=1, keepdim=True)
        tracking_loss = basic.reduce_masked_mean(tracking_loss, valid_g)

        if 'size_bev' in target:
            size_g = target['size_bev']
            rotbin_g = target['rotbin_bev']
            rotres_g = target['rotres_bev']
            size_loss = torch.abs(size_e - size_g).sum(dim=1, keepdim=True)
            size_loss = basic.reduce_masked_mean(size_loss, valid_g)
            rot_loss = compute_rot_loss(rot_e, rotbin_g, rotres_g, valid_g)
        else:
            size_loss = torch.tensor(0.)
            rot_loss = torch.tensor(0.)

        center_factor = 1 / torch.exp(self.model.center_weight)
        center_loss_weight = center_factor * center_loss
        center_uncertainty_loss = self.model.center_weight

        offset_factor = 1 / torch.exp(self.model.offset_weight)
        offset_loss_weight = offset_factor * offset_loss
        offset_uncertainty_loss = self.model.offset_weight

        size_factor = 1 / torch.exp(self.model.size_weight)
        size_loss_weight = size_factor * size_loss
        size_uncertainty_loss = self.model.size_weight

        rot_factor = 1 / torch.exp(self.model.rot_weight)
        rot_loss_weight = rot_factor * rot_loss
        rot_uncertainty_loss = self.model.rot_weight

        tracking_factor = 1 / torch.exp(self.model.tracking_weight)
        tracking_loss_weight = tracking_factor * tracking_loss
        tracking_uncertainty_loss = self.model.tracking_weight

        # img loss
        center_img_loss = self.center_loss_fn(basic.sigmoid(center_img_e), center_img_g) / S

        loss_dict = {
            'center_loss': 10 * center_loss,
            'offset_loss': 10 * offset_loss,
            'tracking_loss': tracking_loss,
            'size_loss': size_loss,
            'rot_loss': rot_loss,
            'center_img': center_img_loss,
        }
        loss_weight_dict = {
            'center_loss': 10 * center_loss_weight,
            'offset_loss': 10 * offset_loss_weight,
            'tracking_loss': tracking_loss_weight,
            'size_loss': size_loss_weight,
            'rot_loss': rot_loss_weight,
            'center_img': center_img_loss,
        }
        stats_dict = {
            'center_uncertainty_loss': center_uncertainty_loss,
            'offset_uncertainty_loss': offset_uncertainty_loss,
            'tracking_uncertainty_loss': tracking_uncertainty_loss,
            'size_uncertainty_loss': size_uncertainty_loss,
            'rot_uncertainty_loss': rot_uncertainty_loss,
        }
        total_loss = sum(loss_weight_dict.values()) + sum(stats_dict.values())

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        total_loss, loss_dict = self.loss(target, output)

        B = item['img'].shape[0]
        self.log('train_loss', total_loss, prog_bar=True, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, batch_size=B)

        return total_loss

    def validation_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        if batch_idx % 100 == 1:
            self.plot_data(target, output, batch_idx)

        total_loss, loss_dict = self.loss(target, output)

        B = item['img'].shape[0]
        self.log('val_loss', total_loss, batch_size=B, sync_dist=True)
        self.log('val_center', loss_dict['center_loss'], batch_size=B, sync_dist=True)
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, batch_size=B, sync_dist=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        item, target = batch
        output = self(item)

        # ref_T_global = item['ref_T_global']
        # global_T_ref = torch.inverse(ref_T_global)

        # output on bev plane
        center_e = output['instance_center']
        offset_e = output['instance_offset']
        size_e = output['instance_size']
        rot_e = output['instance_rot']

        xy_e, xy_prev_e, scores_e, classes_e, sizes_e, rzs_e = decode.decoder(
            center_e.sigmoid(), offset_e, size_e, rz_e=rot_e, K=self.max_detections
        )

        mem_xyz = torch.cat((xy_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy = self.vox_util.Mem2Ref(mem_xyz, self.Y, self.Z, self.X)[..., :2]

        mem_xyz_prev = torch.cat((xy_prev_e, torch.zeros_like(xy_e[..., 0:1])), dim=2)
        ref_xy_prev = self.vox_util.Mem2Ref(mem_xyz_prev, self.Y, self.Z, self.X)[..., :2]

        # detection
        for frame, grid_gt, xy, score in zip(item['frame'], item['grid_gt'], ref_xy, scores_e):
            frame = int(frame.item())
            valid = score > self.conf_threshold

            self.moda_gt_list.extend([[frame, x.item(), y.item()] for x, y, _ in grid_gt[grid_gt.sum(1) != 0]])
            self.moda_pred_list.extend([[frame, x.item(), y.item()] for x, y in xy[valid]])

        # tracking
        for seq_num, frame, grid_gt, bev_det, bev_prev, score, in (
                zip(item['sequence_num'], item['frame'], item['grid_gt'], ref_xy.cpu(), ref_xy_prev.cpu(),
                    scores_e.cpu())):
            frame = int(frame.item())
            output_stracks = self.test_tracker.update(bev_det, bev_prev, score)

            self.mota_gt_list.extend([[seq_num.item(), frame, i.item(), -1, -1, -1, -1, 1, x.item(),  y.item(), -1]
                                      for x, y, i in grid_gt[grid_gt.sum(1) != 0]])
            self.mota_pred_list.extend([[seq_num.item(), frame, s.track_id, -1, -1, -1, -1, s.score.item()]
                                        + s.xy.tolist() + [-1]
                                        for s in output_stracks])

    def on_test_epoch_end(self):
        log_dir = self.trainer.log_dir if self.trainer.log_dir is not None else '../data/cache'

        # detection
        pred_path = osp.join(log_dir, 'moda_pred.txt')
        gt_path = osp.join(log_dir, 'moda_gt.txt')
        np.savetxt(pred_path, np.array(self.moda_pred_list), '%f')
        np.savetxt(gt_path, np.array(self.moda_gt_list), '%d')
        recall, precision, moda, modp = modMetricsCalculator(osp.abspath(pred_path), osp.abspath(gt_path))
        self.log(f'detect/recall', recall)
        self.log(f'detect/precision', precision)
        self.log(f'detect/moda', moda)
        self.log(f'detect/modp', modp)

        # tracking
        scale = 1 if self.X == 150 else 0.025  # HACK
        pred_path = osp.join(log_dir, 'mota_pred.txt')
        gt_path = osp.join(log_dir, 'mota_gt.txt')
        np.savetxt(pred_path, np.array(self.mota_pred_list), '%f', delimiter=',')
        np.savetxt(gt_path, np.array(self.mota_gt_list), '%f', delimiter=',')
        summary = mot_metrics(osp.abspath(pred_path), osp.abspath(gt_path), scale)
        summary = summary.loc['OVERALL']
        for key, value in summary.to_dict().items():
            if value >= 1 and key[:3] != 'num':
                value /= summary.to_dict()['num_unique_objects']
            value = value * 100 if value < 1 else value
            value = 100 - value if key == 'motp' else value
            self.log(f'track/{key}', value)

    def plot_data(self, target, output, batch_idx=0):
        center_e = output['instance_center']
        center_g = target['center_bev']

        # save plots to tensorboard in eval loop
        writer = self.logger.experiment
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.imshow(center_g[-1].amax(0).sigmoid().squeeze().cpu().numpy())
        ax2.imshow(center_e[-1].amax(0).sigmoid().squeeze().cpu().numpy())
        ax1.set_title('center_g')
        ax2.set_title('center_e')
        plt.tight_layout()
        writer.add_figure(f'plot/{batch_idx}', fig, global_step=self.global_step)
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }


if __name__ == '__main__':
    from lightning.pytorch.cli import LightningCLI
    torch.set_float32_matmul_precision('medium')

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.resolution", "data.init_args.resolution")
            parser.link_arguments("model.bounds", "data.init_args.bounds")
            parser.link_arguments("trainer.accumulate_grad_batches", "data.init_args.accumulate_grad_batches")


    cli = MyLightningCLI(WorldTrackModel)
