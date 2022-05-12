import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.layers import ShapeSpec
from wsl.config import add_wsl_config
from detectron2.modeling.backbone import build_backbone
from wsl.modeling.weak_heads import RoIConv

# 首先, RoIConv 这个模块应该是写的没有什么大问题了

# cfg_path = "/home/suil/codes/github/UWSOD/projects/WSL/configs/PascalVOC-Detection/wsod_V16.yaml"
cfg_path = "/home/suil/codes/github/UWSOD/configs/PascalVOC-Detection/retina_V16.yaml"
cfg = get_cfg()
add_wsl_config(cfg)
cfg.merge_from_file(cfg_path)
cfg.freeze()

backbone = build_backbone(cfg)
shapes = backbone.output_shape()
shapes = [shapes[i] for i in shapes]

assert len(shapes) == 1

anchor_generator = build_anchor_generator(cfg, shapes)
# List[Boxes], 其中每一项 Boxes 的 shape 都是 H*W*num_anchors, 4 的
# 这个相当于是 (height: 256, width: 384 的 image)
inputs = torch.Tensor(
    1, 256, 32, 48
)
anchors = anchor_generator(inputs)

anchors = anchors[0].tensor
# [H*W*num_anchors, 4] -> (H*W, num_anchors, 4)
anchors = anchors.view(-1, 18, 4)
anchors1 = anchors[:, 1, :].unsqueeze(0)

roiconv = RoIConv(256, 256, 3)
result, offset = roiconv(inputs.cuda(), anchors1.cuda(), 8, inputs.shape[-2:], test=True)
