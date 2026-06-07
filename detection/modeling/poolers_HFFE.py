# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import sys
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import RoIPool
from typing import List, Optional
from detectron2.layers import ROIAlign, ROIAlignRotated, cat, nonzero_tuple, shapes_to_tensor
from detectron2.structures import Boxes
from detectron2.utils.tracing import assert_fx_safe, is_fx_tracing
import torch.nn.init as init

__all__ = ["ROIPooler"]


@torch.jit.script_if_tracing
def _create_zeros(
        batch_target: Optional[torch.Tensor],
        channels: int,
        height: int,
        width: int,
        like_tensor: torch.Tensor,
) -> torch.Tensor:
    batches = batch_target.shape[0] if batch_target is not None else 0
    sizes = (batches, channels, height, width)
    return torch.zeros(sizes, dtype=like_tensor.dtype, device=like_tensor.device)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
def assign_boxes_to_levels(
        box_lists: List[Boxes],
        min_level: int,
        max_level: int,
        canonical_box_size: int,
        canonical_level: int,
):

    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def convert_boxes_to_pooler_format(box_lists):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """

    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full(
            (len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
        )
        return cat((repeated_index, box_tensor), dim=1)

    pooler_fmt_boxes = cat(
        [fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0.0)

        self.local_att.apply(init_weights)
        self.global_att.apply(init_weights)

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)

        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x * wei + residual * (1 - wei)
        return xo


import torch
from torch import nn
import torch.nn.functional as F


class HFFE_Lite(nn.Module):
    def __init__(self, feature_low_channel, feature_high_channel, out_channel):
        super(HFFE_Lite, self).__init__()

        # 1. 特征图互掩码映射层 (1x1 卷积实现轻量化) [cite: 177]
        self.conv_block_low = nn.Sequential(
            nn.Conv2d(feature_low_channel, 1, 1),
            nn.Sigmoid()
        )
        self.conv_block_high = nn.Sequential(
            nn.Conv2d(feature_high_channel, 1, 1),
            nn.Sigmoid()
        )

        # 2. 统一通道投影层
        self.conv1 = CBR(feature_low_channel, out_channel, 1)
        self.conv2 = CBR(feature_high_channel, out_channel, 1)
        self.conv3 = CBR(feature_low_channel + feature_high_channel, out_channel, 1)

        # 3. 轻量化通道注意力 (SE-Lite) [cite: 134, 203]
        self.ca_lite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // 4, out_channel, 1),
            nn.Sigmoid()
        )

        # 4. 最终融合输出层 [cite: 205]
        self.conv_final = CBR(out_channel * 2, out_channel, 1)

        # 执行权重初始化
        self._init_weights()

    def _init_weights(self):
        """
        专用初始化策略：
        - 卷积层使用 Kaiming Normal 保证深层传播稳定
        - 最终 BN 层初始权重设为 0.1，实现类似残差权重的“渐进式”学习
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print("HFFE")
        # 核心：弱化初期增强路径，保护基础特征 [cite: 157]
        if hasattr(self.conv_final, 'bn'):
            nn.init.constant_(self.conv_final.bn.weight, 0.1)

    def forward(self, x_low, x_high):

        # 空间尺度对齐 (Bilinear Interpolation) [cite: 214]
        print("------------------HFFE Forward Executing --------------------------------")
        if x_low.shape[-2:] != x_high.shape[-2:]:
            x_high = F.interpolate(x_high, size=x_low.shape[-2:], mode='bilinear', align_corners=True)

        # 互掩码生成逻辑：交叉学习跨尺度空间位置感知 [cite: 14, 15]
        low_mask = self.conv_block_low(x_low)
        high_mask = self.conv_block_high(x_high)

        # 特征互感交互 (Mutual Interaction) [cite: 177, 205]
        x_mix = torch.cat([x_low * high_mask, x_high * low_mask], 1)

        # 全局上下文注意力建模 [cite: 14, 203]
        x_context = self.ca_lite(self.conv3(x_mix))

        # 残差式增强特征提取
        x_low_att = x_context * self.conv1(x_low)
        x_high_att = x_context * self.conv2(x_high)

        # 级联输出 [cite: 205, 210]
        out = self.conv_final(torch.cat([x_low_att, x_high_att], 1))
        return out
class CBR(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class AFF_nBN(nn.Module):

    def __init__(self, channels=64, r=4):
        super(AFF_nBN, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0.0)

        self.local_att.apply(init_weights)
        self.global_att.apply(init_weights)

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)

        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x * wei + residual * (1 - wei)
        return xo

        roi_feats_list = []
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            roi_feats_list.append(pooler(x_level, pooler_fmt_boxes))

        """
            multiple AFF
        """

        # aff = AFF(channels=num_channels).to(device)
        # feature_12 = aff(roi_feats_list[0], roi_feats_list[1])
        # feature_123 = aff(feature_12, roi_feats_list[2])
        # feature_1234 = aff(feature_123, roi_feats_list[3])
        # output = feature_1234
        # print(output.shape)
        # exit(0)
        # return output

        # # concat in channel dims [batchsize, NC, h, w]
        # concat_roi_feats = torch.cat(roi_feats_list, dim=1)
        # # [batchszie, N, h, w]
        # spatial_attention_map = self.spatial_attention_conv(concat_roi_feats)

        # for i in range(self.canonical_level):
        #     output += (F.sigmoid(spatial_attention_map[:, i, None, :, :]) * roi_feats_list[i])
        # print(output.shape)
        # exit(0)
        # return output


class ROIPooler_aug(nn.Module):


    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            pooler_type,
            canonical_box_size=224,
            canonical_level=4,
            out_channels=256,
    ):

        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        self.hffe = HFFE_Lite(feature_low_channel=out_channels,
                              feature_high_channel=out_channels,
                              out_channel=out_channels)

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
                len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
        # input_size: concat_channel
        self.spatial_attention_conv = nn.Sequential(nn.Conv2d(out_channels * canonical_level, out_channels, 1),
                                                    nn.ReLU(),
                                                    nn.Conv2d(out_channels, canonical_level, 3, padding=1))

    def forward(self, x: List[torch.Tensor], box_lists):

        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
                len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        # add batch index to every box_tensor, and flatten all box is  relatively to eatch orign img
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        # low level x: [level, NCHW]
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        # assign each box to a level, [N], N is the number of boxes
        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )
        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])
        # aff.eval()
        # 假设 HFFE_Lite 已经按照之前的轻量化方案定义
        # self.hffe = HFFE_Lite(256, 256, 256)

        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            if len(inds) == 0: continue  # 性能优化：无目标时跳过

            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            width, height = x[level].shape[2], x[level].shape[3]

            # 尺寸对齐 (ACFF 策略) [cite: 208, 214]
            # 使用 align_corners=False 配合 mode='bilinear' 通常在检测任务中更稳定
            f1 = F.interpolate(x[(level + 1) % 4], size=(width, height), mode='bilinear', align_corners=False)
            f2 = F.interpolate(x[(level + 2) % 4], size=(width, height), mode='bilinear', align_corners=False)
            f3 = F.interpolate(x[(level + 3) % 4], size=(width, height), mode='bilinear', align_corners=False)

            # 链式 HFFE-Lite 融合
            # 每一层输出的通道数均为 256，与下一层输入对齐
            f12 = self.hffe(x[level], f1)
            f123 = self.hffe(f12, f2)
            f1234 = self.hffe(f123, f3)

            # 这里的 f1234 是最终增强后的全尺度特征图
            # pooler 负责执行 ROIAlign，得到最终用于分类或得分头的池化特征 [cite: 163, 188]
            output.index_put_((inds,), pooler(f1234, pooler_fmt_boxes_level))
        return output

        roi_feats_list = []
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            roi_feats_list.append(pooler(x_level, pooler_fmt_boxes))

        """
            multiple AFF
        """

        # aff = AFF(channels=num_channels).to(device)
        # feature_12 = aff(roi_feats_list[0], roi_feats_list[1])
        # feature_123 = aff(feature_12, roi_feats_list[2])
        # feature_1234 = aff(feature_123, roi_feats_list[3])
        # output = feature_1234
        # print(output.shape)
        # exit(0)
        # return output

        # # concat in channel dims [batchsize, NC, h, w]
        # concat_roi_feats = torch.cat(roi_feats_list, dim=1)
        # # [batchszie, N, h, w]
        # spatial_attention_map = self.spatial_attention_conv(concat_roi_feats)

        # for i in range(self.canonical_level):
        #     output += (F.sigmoid(spatial_attention_map[:, i, None, :, :]) * roi_feats_list[i])
        # print(output.shape)
        # exit(0)
        # return output


class ROIPooler_aug_nBN(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            pooler_type,
            canonical_box_size=224,
            canonical_level=4,
            out_channels=256,
    ):

        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        self.aff = AFF_nBN(channels=out_channels)

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
                len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
        # input_size: concat_channel
        self.spatial_attention_conv = nn.Sequential(nn.Conv2d(out_channels * canonical_level, out_channels, 1),
                                                    nn.ReLU(),
                                                    nn.Conv2d(out_channels, canonical_level, 3, padding=1))

    def forward(self, x: List[torch.Tensor], box_lists):

        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):  [514, 4]
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
                len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        # add batch index to every box_tensor, and flatten all box is  relatively to eatch orign img
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        # low level x: [level, NCHW]
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        # assign each box to a level, [N], N is the number of boxes
        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )
        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])
        # aff.eval()
        for level, pooler in enumerate(self.level_poolers):
            # find boxes belong to this level
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            width, height = x[level].shape[2], x[level].shape[3]
            feature_1 = F.interpolate(x[(level + 1) % 4].clone(), size=(width, height), mode='bilinear')
            feature_2 = F.interpolate(x[(level + 2) % 4].clone(), size=(width, height), mode='bilinear')
            feature_3 = F.interpolate(x[(level + 3) % 4].clone(), size=(width, height), mode='bilinear')
            feature_12 = self.aff(x[level], feature_1)
            feature_123 = self.aff(feature_12, feature_2)
            feature_1234 = self.aff(feature_123, feature_3)
            output.index_put_((inds,), pooler(feature_1234, pooler_fmt_boxes_level))
        return output


class ROIPooler_TwoFusion(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
            self,
            output_size,
            scales,
            sampling_ratio,
            pooler_type,
            canonical_box_size=224,
            canonical_level=4,
            out_channels=256,
    ):

        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        self.aff = AFF(channels=out_channels)

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
                len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
        # input_size: concat_channel
        self.spatial_attention_conv = nn.Sequential(nn.Conv2d(out_channels * canonical_level, out_channels, 1),
                                                    nn.ReLU(),
                                                    nn.Conv2d(out_channels, canonical_level, 3, padding=1))

    def forward(self, x: List[torch.Tensor], box_lists):

        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):  [514, 4]
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
                len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        # add batch index to every box_tensor, and flatten all box is  relatively to eatch orign img
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        # low level x: [level, NCHW]
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        # assign each box to a level, [N], N is the number of boxes
        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )
        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])
        # aff.eval()
        for level, pooler in enumerate(self.level_poolers):
            # find boxes belong to this level
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            width, height = x[level].shape[2], x[level].shape[3]
            feature_1 = F.interpolate(x[(level + 1) % 4].clone(), size=(width, height), mode='bilinear')
            feature_12 = self.aff(x[level], feature_1)
            output.index_put_((inds,), pooler(feature_12, pooler_fmt_boxes_level))
        return output