import torch
import torch.nn as nn

from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers


class MyNewClassHead(FastRCNNOutputLayers):
   

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        input_size = self.cls_score.in_features
        hidden_dim = max(input_size // 4, 64)

        
        self.cls_adapter = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_size),
        )

        nn.init.zeros_(self.cls_adapter[-1].weight)
        nn.init.zeros_(self.cls_adapter[-1].bias)

        
        self.res_scale = 0.01

        
        self.positive_scale = 0.2
        print("0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 ")

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

       
        proposal_deltas = self.bbox_pred(x)

        
        base_scores = self.cls_score(x)
        base_known = base_scores[:, :-1]
        base_bg = base_scores[:, -1:]

        
        delta = self.cls_adapter(x)
        
        cls_x = x + self.res_scale * delta

        adapt_scores = self.cls_score(cls_x)
        adapt_known = adapt_scores[:, :-1]

        
        logit_delta = adapt_known - base_known

        
        negative_delta = logit_delta.clamp(max=0.0)
        positive_delta = logit_delta.clamp(min=0.0)

        final_known = base_known + negative_delta + self.positive_scale * positive_delta

        
        final_scores = torch.cat([final_known, base_bg], dim=1)

        return final_scores, proposal_deltas
