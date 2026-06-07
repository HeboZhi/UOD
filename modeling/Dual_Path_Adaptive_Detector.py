from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage


import torch
import torch.nn as nn
import torch.nn.functional as F

class DPAD(nn.Module):
    def __init__(self, dim: int, num_prototypes: int = 5, detach_features: bool = True):
        super().__init__()
        self.dim = dim
        self.N = num_prototypes
        self.detach_features = detach_features

        self.prototypes = nn.Parameter(torch.randn(self.N, dim))
        nn.init.orthogonal_(self.prototypes)

        self.norm = nn.LayerNorm(dim)

        self.feature_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim)
        )

        nn.init.constant_(self.feature_gate[-1].weight, 0)
        nn.init.constant_(self.feature_gate[-1].bias, 0)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))
        self.bias = nn.Parameter(torch.tensor([2.0]))

    def forward(self, box_features: torch.Tensor, return_prob: bool = True):
        feats = box_features.detach() if self.detach_features else box_features

        base_feat = self.norm(feats)
        offset = self.feature_gate(base_feat)
        proj = base_feat + offset

        proj = F.normalize(proj, dim=-1)
        norm_proto = F.normalize(self.prototypes, dim=-1)
        sims = proj @ norm_proto.T

        if self.N >= 2:
            max_val, _ = torch.topk(sims, k=2, dim=1)
            max_val = max_val.mean(dim=1)
        else:
            max_val = sims.max(dim=1).values

        scale = self.logit_scale.exp()
        logits = scale * max_val + self.bias

        if return_prob:
            return torch.sigmoid(logits).unsqueeze(1)
        return logits.unsqueeze(1)
