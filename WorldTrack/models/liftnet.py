# 导入 NumPy 库，用于数值计算
import numpy as np
# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn

# 导入基础工具函数（用于张量的打包/解包等操作）
import utils.basic
# 导入几何变换工具（用于坐标系转换、投影等）
import utils.geom
# 导入体素化工具（用于将2D特征提升到3D空间）
import utils.vox
# 导入解码器模块（用于从BEV特征解码出检测结果）
from models.decoder import Decoder
# 导入多种编码器（ResNet101, ResNet50, EfficientNet, Swin Transformer, ResNet18）
from models.encoder import Encoder_res101, Encoder_res50, Encoder_eff, Encoder_swin_t, Encoder_res18


# Liftnet 类：多视角3D目标检测网络，将2D图像特征提升到3D BEV空间
class Liftnet(nn.Module):
    def __init__(self, Y, Z, X,  # Y, Z, X: BEV网格的尺寸（Y=高度维度, Z=垂直维度, X=宽度维度）
                 DMAX,  # 深度范围的最大值（相机可以看到的最远距离）
                 D,  # 深度离散化的数量（将连续深度离散化为 D 个bin）
                 DMIN=2.0,  # 深度范围的最小值（相机的最近可视距离，默认2米）
                 num_classes=None,  # 检测的目标类别数量
                 num_cameras=None,  # 相机数量（如果指定则使用特定的多相机融合策略）
                 do_rgbcompress=True,  # 是否对RGB特征进行压缩
                 rand_flip=False,  # 是否使用随机翻转数据增强
                 latent_dim=256,  # BEV特征的潜在维度
                 feat2d_dim=96,  # 2D特征的维度
                 encoder_type='swin_t',  # 编码器类型（支持多种backbone）
                 z_sign=1,  # Z轴方向的符号（1或-1，用于不同坐标系）
                 ):
        # 调用父类的初始化方法
        super(Liftnet, self).__init__()
        # 断言确保编码器类型在支持的列表中
        assert (encoder_type in ['res101', 'res50', 'res18', 'effb0', 'effb4', 'swin_t'])

        # 保存BEV网格的尺寸参数
        self.Y, self.Z, self.X = Y, Z, X
        # 保存深度范围的最大值和最小值
        self.DMAX, self.DMIN = DMAX, DMIN
        # 保存深度离散化的数量
        self.D = D
        # 保存是否进行RGB压缩的标志
        self.do_rgbcompress = do_rgbcompress
        # 保存是否使用随机翻转的标志
        self.rand_flip = rand_flip
        # 保存潜在特征维度
        self.latent_dim = latent_dim
        # 保存编码器类型
        self.encoder_type = encoder_type
        # 保存相机数量
        self.num_cameras = num_cameras
        # 保存Z轴符号
        self.z_sign = z_sign

        # ImageNet 数据集的均值，用于图像归一化（RGB三通道）
        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).float().cuda()
        # ImageNet 数据集的标准差，用于图像归一化（RGB三通道）
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).float().cuda()

        # ==================== 编码器初始化 ====================
        # 保存2D特征维度
        self.feat2d_dim = feat2d_dim
        # 根据指定的编码器类型初始化对应的编码器
        # 注意：编码器输出通道数 = feat2d_dim（特征维度）+ D（深度维度）
        if encoder_type == 'res101':
            # ResNet-101 编码器
            self.encoder = Encoder_res101(feat2d_dim + self.D)
        elif encoder_type == 'res50':
            # ResNet-50 编码器
            self.encoder = Encoder_res50(feat2d_dim + self.D)
        elif encoder_type == 'res18':
            # ResNet-18 编码器（轻量级）
            self.encoder = Encoder_res18(feat2d_dim + self.D)
        elif encoder_type == 'effb0':
            # EfficientNet-B0 编码器
            self.encoder = Encoder_eff(feat2d_dim + self.D, version='b0')
        elif encoder_type == 'swin_t':
            # Swin Transformer Tiny 编码器
            self.encoder = Encoder_swin_t(feat2d_dim + self.D)
        else:
            # 默认使用 EfficientNet-B4 编码器
            self.encoder = Encoder_eff(feat2d_dim + self.D, version='b4')

        # ==================== BEV 特征压缩器 ====================
        # 如果指定了相机数量，则创建相机特征压缩器
        if self.num_cameras is not None:
            # 相机压缩器：将多个相机的3D特征融合为单一的3D特征
            self.cam_compressor = nn.Sequential(
                # 3D卷积：输入通道 = feat2d_dim * num_cameras，输出通道 = feat2d_dim
                nn.Conv3d(feat2d_dim * self.num_cameras, feat2d_dim, kernel_size=3, padding=1, stride=1),
                # 实例归一化 + ReLU 激活
                nn.InstanceNorm3d(feat2d_dim), nn.ReLU(),
                # 1x1x1 卷积进行特征精炼
                nn.Conv3d(feat2d_dim, feat2d_dim, kernel_size=1),
            )

        # BEV特征压缩器：将3D体素特征压缩为2D BEV特征
        self.bev_compressor = nn.Sequential(
            # 2D卷积：输入 = feat2d_dim * Z（沿Z轴展平），输出 = latent_dim
            nn.Conv2d(self.feat2d_dim * self.Z, latent_dim, kernel_size=3, padding=1),
            # 实例归一化 + ReLU 激活
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            # 1x1 卷积进行特征精炼
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

        # BEV时序融合模块：融合当前帧和前一帧的BEV特征
        self.bev_temporal = nn.Sequential(
            # 2D卷积：输入 = latent_dim * 2（当前帧+前一帧拼接），输出 = latent_dim
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            # 实例归一化 + ReLU 激活
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            # 1x1 卷积进行特征精炼
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

        # ==================== 解码器 ====================
        # 从BEV特征解码出目标检测结果（中心点、偏移、尺寸、旋转等）
        self.decoder = Decoder(
            in_channels=latent_dim,  # 输入BEV特征的通道数
            n_classes=num_classes,  # 目标类别数
            feat2d=feat2d_dim,  # 2D特征维度
        )

        # ==================== 可学习的损失权重 ====================
        # 使用 nn.Parameter 使这些权重可以在训练过程中自动学习和调整
        # 中心点检测的损失权重
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # 偏移量预测的损失权重
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # 跟踪ID的损失权重（用于多目标跟踪）
        self.tracking_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # 目标尺寸预测的损失权重
        self.size_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # 目标旋转角度预测的损失权重
        self.rot_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def create_frustum(self):
        """
        创建视锥体（Frustum）网格
        视锥体表示相机的可视范围，用于将图像平面的像素坐标与3D空间关联
        """
        # 获取图像的最终尺寸（经过数据增强后）
        ogfH, ogfW = self.data_aug_conf['final_dim']
        # 计算下采样后的特征图尺寸
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        # 创建深度采样点：从最小深度到最大深度均匀采样 D 个点
        # 形状: (D, fH, fW)，每个位置都有 D 个深度值
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        # 获取深度维度大小
        D, _, _ = ds.shape
        # 创建 X 坐标网格（宽度方向），从 0 到 ogfW-1
        # 形状: (D, fH, fW)
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        # 创建 Y 坐标网格（高度方向），从 0 到 ogfH-1
        # 形状: (D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # 将 X, Y, 深度堆叠成视锥体网格
        # 形状: (D, H, W, 3)，最后一维是 [x, y, depth]
        frustum = torch.stack((xs, ys, ds), -1)
        # 返回为不可训练的参数（固定的几何结构）
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        获取几何坐标变换
        将图像平面上的点云转换到自车（ego）坐标系中的 (x,y,z) 位置
        
        参数:
            rots: 旋转矩阵（相机到世界坐标系）
            trans: 平移向量（相机到世界坐标系）
            intrins: 相机内参矩阵
            post_rots: 数据增强后的旋转矩阵
            post_trans: 数据增强后的平移向量
        
        返回: 
            points: 形状 (B, N, D, H/downsample, W/downsample, 3)
                   表示每个像素在不同深度下的3D坐标
        """
        # 获取批次大小和相机数量
        B, N, _ = trans.shape

        # ==================== 步骤1：撤销数据增强的变换 ====================
        # 减去平移并应用旋转的逆变换，恢复到原始图像坐标
        # 形状: (B, N, D, H, W, 3)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # ==================== 步骤2：从相机坐标系转换到自车坐标系 ====================
        # 将像素坐标 (u, v, d) 转换为 3D 坐标 (x, y, z)
        # 前两维乘以深度值（透视投影的逆操作）
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]  # 保持深度值
                            ), 5)
        # 组合旋转矩阵和内参矩阵的逆
        combine = rots.matmul(torch.inverse(intrins))
        # 应用组合变换矩阵
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # 加上相机的世界坐标位置（平移）
        points += trans.view(B, N, 1, 1, 1, 3)

        # 返回自车坐标系下的3D点云
        return points

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global, prev_bev=None):
        """
        前向传播：将多视角RGB图像转换为BEV特征，并预测目标检测结果
        
        参数说明:
            B = 批次大小, S = 相机数量, C = 3（RGB通道）, H = 图像高度, W = 图像宽度
            rgb_cams: 多视角RGB图像，形状 (B, S, C, H, W)
            pix_T_cams: 像素到相机坐标的变换矩阵，形状 (B, S, 4, 4)
            cams_T_global: 相机到全局坐标的变换矩阵，形状 (B, S, 4, 4)
            vox_util: 体素化工具对象（用于3D网格操作）
            ref_T_global: 参考坐标系到全局坐标的变换矩阵，形状 (B, 4, 4)
            prev_bev: 前一帧的BEV特征（用于时序融合），可选
        
        返回:
            out_dict: 包含检测结果的字典（中心点、偏移、尺寸、旋转、深度等）
        """
        # 获取输入张量的维度
        B, S, C, H, W = rgb_cams.shape
        # 确保输入是RGB图像（3通道）
        assert (C == 3)
        
        # ==================== 步骤1：张量重塑和坐标变换准备 ====================
        # 定义辅助函数：将序列维度（相机维度）打包到批次维度中
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        # 定义辅助函数：将打包的维度恢复
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        
        # 将所有相机的图像和变换矩阵打包：(B, S, ...) -> (B*S, ...)
        rgb_cams_ = __p(rgb_cams)  # (B*S, C, H, W)
        pix_T_cams_ = __p(pix_T_cams)  # (B*S, 4, 4)
        cams_T_global_ = __p(cams_T_global)  # (B*S, 4, 4) 相机到全局坐标
        global_T_cams_ = torch.inverse(cams_T_global_)  # (B*S, 4, 4) 全局到相机坐标
        ref_T_cams = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # (B*S, 4, 4) 参考系到相机
        cams_T_ref_ = torch.inverse(ref_T_cams)  # (B*S, 4, 4) 相机到参考系

        # ==================== 步骤2：RGB图像编码 ====================
        # 获取设备信息
        device = rgb_cams_.device
        # 使用 ImageNet 均值和标准差对图像进行归一化
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)
        
        # 如果启用随机翻转数据增强
        if self.rand_flip:
            B0, _, _, _ = rgb_cams_.shape
            # 随机选择一部分样本进行水平翻转
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            rgb_cams_[self.rgb_flip_index] = torch.flip(rgb_cams_[self.rgb_flip_index], [-1])
        
        # 通过编码器提取2D特征
        # 输出形状: (B*S, D+feat2d_dim, H/8, W/8)
        # 前 D 个通道是深度预测，后 feat2d_dim 个通道是图像特征
        feat_cams_ = self.encoder(rgb_cams_)
        
        # 如果进行了翻转，需要将特征也翻转回来
        if self.rand_flip:
            feat_cams_[self.rgb_flip_index] = torch.flip(feat_cams_[self.rgb_flip_index], [-1])
        
        # 获取特征图的尺寸
        _, CD, Hf, Wf = feat_cams_.shape
        # 验证通道数（已注释，但通常 CD = D + feat2d_dim）
        # assert (CD == (self.D + self.feat2d_dim))

        # 计算特征图相对于原图的缩放比例
        sy = Hf / float(H)  # 高度方向的缩放
        sx = Wf / float(W)  # 宽度方向的缩放
        # 获取BEV网格尺寸
        Y, Z, X = self.Y, self.Z, self.X
        # 根据缩放比例调整相机内参矩阵
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)

        # ==================== 步骤3：分离深度预测和图像特征 ====================
        # 提取深度预测通道（前 D 个通道）
        # 形状: (B*S, 1, D, Hf, Wf) - 每个像素位置预测 D 个深度的logits
        depth_cams_out = feat_cams_[:, :self.D].unsqueeze(1)
        
        # 提取图像特征通道（后 feat2d_dim 个通道）
        # 形状: (B*S, feat2d_dim, 1, Hf, Wf)
        feat_cams_ = feat_cams_[:, self.D:].unsqueeze(2)
        
        # 对深度logits应用softmax，得到深度概率分布
        # 形状: (B*S, 1, D, Hf, Wf) - 每个像素的深度分布（和为1）
        depth_cams_ = depth_cams_out.softmax(dim=2)
        
        # 将图像特征与深度分布相乘，得到深度加权的特征
        # 形状: (B*S, feat2d_dim, D, Hf, Wf)
        # 这是 "Lifting" 的关键步骤：将 2D 特征扩展到多个深度平面
        feat_tileXs_ = feat_cams_ * depth_cams_

        # ==================== 步骤4：将2D特征反投影到3D体素网格 ====================
        # 使用体素化工具将深度加权的图像特征映射到3D空间
        # 输入: (B*S, feat2d_dim, D, Hf, Wf)
        # 输出: (B*S, feat2d_dim, Y, Z, X) - 3D体素特征
        feat_mems_ = vox_util.warp_tiled_to_mem(
            feat_tileXs_,  # 深度加权的2D特征
            utils.basic.matmul2(featpix_T_cams_, cams_T_ref_),  # 特征像素到参考系的变换
            cams_T_ref_,  # 相机到参考系的变换
            Y, Z, X,  # 3D网格尺寸
            self.DMIN, self.DMAX+self.DMIN,  # 深度范围
            z_sign=self.z_sign)  # Z轴方向
        
        # 恢复批次和相机维度: (B*S, ...) -> (B, S, ...)
        feat_mems = __u(feat_mems_)

        # ==================== 步骤5：多相机特征融合 ====================
        if self.num_cameras is None:
            # 方式1：使用加权平均融合多个相机的特征
            # 创建权重掩码（记录每个体素被多少个相机观测到）
            one_mems_ = vox_util.warp_tiled_to_mem(
                torch.ones_like(feat_tileXs_),  # 全1张量
                utils.basic.matmul2(featpix_T_cams_, cams_T_ref_),
                cams_T_ref_, Y, Z, X, self.DMIN, self.DMAX+self.DMIN, z_sign=self.z_sign)
            one_mems = __u(one_mems_)
            # 防止除零：将权重最小值设为1
            one_mems = one_mems.clamp(min=1.0)

            # 计算加权平均：对相机维度求和后除以权重
            # 输出形状: (B, feat2d_dim, Y, Z, X)
            feat_mem = utils.basic.reduce_masked_mean(feat_mems, one_mems, dim=1)
        else:
            # 方式2：使用3D卷积网络融合多个相机的特征
            # 先展平相机和特征维度: (B, S, C, Y, Z, X) -> (B, S*C, Y, Z, X)
            # 输出形状: (B, feat2d_dim, Y, Z, X)
            feat_mem = self.cam_compressor(feat_mems.flatten(1, 2))

        # ==================== 步骤6：BEV数据增强（随机翻转）====================
        if self.rand_flip:
            # 随机选择一部分样本在两个方向上进行翻转
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            # 沿 X 轴翻转（最后一个维度）
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            # 沿 Y 轴翻转（倒数第三个维度）
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

        # ==================== 步骤7：3D特征压缩为2D BEV特征 ====================
        # 重排维度并展平Z轴：(B, C, Y, Z, X) -> (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = feat_mem.permute(0, 1, 3, 2, 4).flatten(1, 2)
        # 通过2D卷积压缩特征：(B, C*Z, Y, X) -> (B, latent_dim, Y, X)
        bev_features = self.bev_compressor(bev_features)

        # ==================== 步骤8：时序特征融合 ====================
        # 如果没有提供前一帧的BEV特征，使用当前帧代替
        if prev_bev is None:
            prev_bev = bev_features
        # 拼接当前帧和前一帧的BEV特征：(B, latent_dim*2, Y, X)
        bev_features = torch.cat([bev_features, prev_bev], dim=1)
        # 通过时序融合网络处理：(B, latent_dim*2, Y, X) -> (B, latent_dim, Y, X)
        bev_features = self.bev_temporal(bev_features)

        # ==================== 步骤9：BEV解码器 - 预测检测结果 ====================
        # 从BEV特征解码出目标检测的各种属性
        # 输入:
        #   - bev_features: BEV特征，形状 (B, latent_dim, Y, X)
        #   - feat_cams_.squeeze(2): 2D图像特征（用于某些解码器架构）
        #   - 翻转索引（如果启用了数据增强，用于翻转回预测结果）
        # 输出: out_dict 包含以下内容：
        #   - 'center': 目标中心点热图
        #   - 'offset': 亚像素偏移
        #   - 'size': 目标尺寸（长、宽、高）
        #   - 'rot': 目标旋转角度
        #   - 'tracking': 跟踪嵌入向量（用于多目标跟踪）
        out_dict = self.decoder(bev_features, feat_cams_.squeeze(2),
                                (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
        
        # 将深度预测结果也添加到输出字典中
        # 用于监督学习或可视化
        out_dict['depth'] = depth_cams_out

        # 返回包含所有预测结果的字典
        return out_dict
