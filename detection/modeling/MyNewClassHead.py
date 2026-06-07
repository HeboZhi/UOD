import torch
import torch.nn as nn

from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


class MyNewClassHead(FastRCNNOutputLayers):
    """
    KNA Head: Known-logit Non-Amplifying Classification Head

    设计目标：
    1. 保留原始 cls_score；
    2. 保留原始 bbox_pred；
    3. 保留原始 0.01 residual adapter；
    4. 不改 DPAD；
    5. 不增加复杂分支；
    6. 不做 top1 定向抑制；
    7. 只限制 adapter 对 known logits 的正向放大，
       避免纯未知目标被已知分类头额外推成 known 类。
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        input_size = self.cls_score.in_features
        hidden_dim = max(input_size // 4, 64)

        # 原始 0.01 residual adapter
        self.cls_adapter = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_size),
        )

        nn.init.zeros_(self.cls_adapter[-1].weight)
        nn.init.zeros_(self.cls_adapter[-1].bias)

        # 保留原始 0.01
        self.res_scale = 0.01

        # 只保留 adapter 正向 known-logit 提升的一小部分
        # 0.0 = 完全不允许 adapter 提高 known logits
        # 0.2 = 只允许 20% 正向提升
        # 1.0 = 等价原始 0.01 head
        self.positive_scale = 0.2
        print("0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 ")

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # bbox 分支完全不动
        proposal_deltas = self.bbox_pred(x)

        # baseline known classifier anchor
        base_scores = self.cls_score(x)
        base_known = base_scores[:, :-1]
        base_bg = base_scores[:, -1:]

        # 原始 0.01 residual adapter
        delta = self.cls_adapter(x)
        #  cls_x =x+0.01x
        cls_x = x + self.res_scale * delta

        adapt_scores = self.cls_score(cls_x)
        adapt_known = adapt_scores[:, :-1]

        # adapter 引起的 known-logit 漂移
        logit_delta = adapt_known - base_known

        # 非对称约束：
        # 负向漂移保留，允许降低错误 known 激活；
        # 正向漂移只保留很小比例，避免把 unknown 推成 known。
        negative_delta = logit_delta.clamp(max=0.0)
        positive_delta = logit_delta.clamp(min=0.0)

        final_known = base_known + negative_delta + self.positive_scale * positive_delta

        # background 用 base_bg，避免 adapter 改变前景/背景边界
        final_scores = torch.cat([final_known, base_bg], dim=1)

        return final_scores, proposal_deltas