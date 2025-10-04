# ==================== 损失函数模块 ====================
# 本文件定义了多种损失函数，用于训练多视角3D目标检测模型
# 主要包括：Focal Loss（中心点检测）、旋转损失、平衡MSE损失等

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import basic


class SimpleLoss(torch.nn.Module):
    """
    简单的带权重的二元交叉熵损失
    
    用于处理正负样本不平衡的问题
    """
    def __init__(self, pos_weight):
        """
        参数:
            pos_weight: 正样本的权重，用于平衡正负样本
        """
        super(SimpleLoss, self).__init__()
        # 使用带权重的二元交叉熵损失，不进行reduction
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        """
        前向传播
        
        参数:
            ypred: 预测值（logits）
            ytgt: 目标值（0或1）
            valid: 有效位置掩码
        
        返回:
            loss: 掩码平均损失
        """
        # 计算逐元素损失
        loss = self.loss_fn(ypred, ytgt)
        # 使用掩码计算平均损失（只考虑有效位置）
        loss = basic.reduce_masked_mean(loss, valid)
        return loss


class FocalLoss(torch.nn.Module):
    """
    Focal Loss - 用于解决目标检测中的类别不平衡问题
    
    来源：Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet)
    特别适用于中心点检测任务，能够降低易分类样本的权重，
    让模型更关注难分类的样本
    """

    def __init__(self, use_distance_weight=False):
        """
        参数:
            use_distance_weight: 是否使用距离权重（给图像中心更高权重）
        """
        super(FocalLoss, self).__init__()
        self.use_distance_weight = use_distance_weight

    def forward(self, pred, gt):
        """
        Focal Loss 前向传播
        
        修改版的 Focal Loss，与 CornerNet 完全相同
        运行更快但占用稍多内存
        
        参数:
            pred: 预测概率 (batch x c x h x w)，已经过sigmoid
            gt: 目标热图 (batch x c x h x w)，值在[0,1]之间
        
        返回:
            loss: Focal Loss值
        """
        # ==================== 步骤1：找出正样本和负样本位置 ====================
        pos_inds = gt.eq(1).float()  # 正样本掩码（gt=1的位置）
        neg_inds = gt.lt(1).float()  # 负样本掩码（gt<1的位置）

        # ==================== 步骤2：计算距离权重（可选）====================
        distance_weight = torch.ones_like(gt)
        if self.use_distance_weight:
            # 为图像中心区域赋予更高权重
            w, h = gt.shape[-2:]
            xs = torch.linspace(-1, 1, steps=h, device=gt.device)
            ys = torch.linspace(-1, 1, steps=w, device=gt.device)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            # 使用正弦函数创建距离权重：中心权重高，边缘权重低
            distance_weight = 9 * torch.sin(torch.sqrt(x * x + y * y)) + 1

        # ==================== 步骤3：计算负样本权重 ====================
        # 论文中的参数：alpha=2, beta=4
        # 对于负样本，gt值越接近1（即越难区分），权重越高
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # ==================== 步骤4：计算正样本和负样本的损失 ====================
        # 正样本损失：-log(p) * (1-p)^2 * 正样本掩码 * 距离权重
        # (1-p)^2 是调制因子，当预测概率p接近1时，损失权重降低
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * distance_weight
        
        # 负样本损失：-log(1-p) * p^2 * 负样本权重 * 负样本掩码 * 距离权重
        # p^2 是调制因子，当预测概率p接近0时，损失权重降低
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * distance_weight

        # ==================== 步骤5：归一化损失 ====================
        num_pos = pos_inds.sum()  # 正样本数量
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            # 如果没有正样本，只使用负样本损失
            loss = loss - neg_loss
        else:
            # 按正样本数量归一化总损失
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


def balanced_mse_loss(pred, gt, valid=None):
    """
    平衡的均方误差损失
    
    分别计算正样本和负样本的MSE损失，然后取平均
    用于处理正负样本不平衡的回归问题
    
    参数:
        pred: 预测值
        gt: 目标值
        valid: 有效位置掩码（可选）
    
    返回:
        loss: 平衡MSE损失
    """
    # 根据目标值划分正负样本
    pos_mask = gt.gt(0.5).float()  # 正样本掩码
    neg_mask = gt.lt(0.5).float()  # 负样本掩码
    
    if valid is None:
        valid = torch.ones_like(pos_mask)
    
    # 计算逐元素MSE损失
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    
    # 分别计算正负样本的平均损失
    pos_loss = basic.reduce_masked_mean(mse_loss, pos_mask * valid)
    neg_loss = basic.reduce_masked_mean(mse_loss, neg_mask * valid)
    
    # 取平均作为最终损失
    loss = (pos_loss + neg_loss) * 0.5

    return loss


class BinRotLoss(nn.Module):
    """
    二进制旋转损失的包装类
    
    用于目标的旋转角度预测
    """
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, rotbin, rotres):
        """
        参数:
            output: 模型输出的旋转预测
            mask: 有效位置掩码
            rotbin: 旋转角度的二进制分类目标
            rotres: 旋转角度的回归残差目标
        """
        loss = compute_rot_loss(output, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    """
    计算回归残差损失（Smooth L1 Loss）
    
    参数:
        output: 预测的残差
        target: 目标残差
    
    返回:
        loss: Smooth L1损失
    """
    return F.smooth_l1_loss(output, target, reduction='mean')


def compute_bin_loss(output, target, mask):
    """
    计算二进制分类损失
    
    参数:
        output: 预测的分类logits
        target: 目标类别
        mask: 有效位置掩码
    
    返回:
        loss: 交叉熵损失
    """
    # 应用掩码到输出
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    """
    计算旋转角度损失
    
    使用双bin策略预测旋转角度：
    - 每个bin负责180度范围内的角度预测
    - 每个bin包含分类损失（是否属于该bin）和回归损失（具体角度）
    
    参数:
        output: 模型输出 (B, 128, 8)
                格式：[bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
                      bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
        target_bin: 目标bin分类 (B, 128, 2) [bin1_cls, bin2_cls]
        target_res: 目标角度残差 (B, 128, 2) [bin1_res, bin2_res]
        mask: 有效位置掩码 (B, 128, 1)
    
    返回:
        loss: 总的旋转损失（分类损失 + 回归损失）
    """
    # 重塑张量为2D形式便于计算
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    
    # ==================== 计算分类损失 ====================
    # Bin1 的分类损失（判断角度是否属于bin1）
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    # Bin2 的分类损失（判断角度是否属于bin2）
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    
    # ==================== 计算回归损失 ====================
    loss_res = torch.zeros_like(loss_bin1)
    
    # 对于属于bin1的样本，计算sin和cos的回归损失
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]  # 找出属于bin1的样本索引
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        
        # 计算sin分量的损失
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        # 计算cos分量的损失
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    
    # 对于属于bin2的样本，计算sin和cos的回归损失
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]  # 找出属于bin2的样本索引
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        
        # 计算sin分量的损失
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        # 计算cos分量的损失
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    
    # 返回总损失：分类损失 + 回归损失
    return loss_bin1 + loss_bin2 + loss_res