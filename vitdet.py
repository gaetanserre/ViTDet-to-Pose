#
# Created in 2022 by Gaëtan Serré
#

import torchvision
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
import torch.nn as nn
from copy import deepcopy

from detectron2.modeling.backbone.vit import ViT
from detectron2.modeling.backbone.vit import SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.layers.batch_norm import LayerNorm

"""
In this file, we instantiate the human-pose ViTDet model
either from scratch or from the pre-trained pre-trained model.
We start by recreating the ViTDet architecture in pure PyTorch
and then we replace the backbone by the ViTDet one.
"""

def get_original_vitdet(pre_trained):
  from detectron2.config import LazyConfig, instantiate

  from detectron2.engine import default_setup

  from detectron2.engine.defaults import create_ddp_model

  from detectron2.checkpoint import DetectionCheckpointer

  cfg = LazyConfig.load("weights/vit_weights/ViTDet_config.py")

  import torch as th

  default_setup(cfg, None)

  ViTDet = instantiate(cfg.model)

  ViTDet = create_ddp_model(ViTDet)

  if pre_trained:
    DetectionCheckpointer(ViTDet).load("weights/vit_weights/model_final_61ccd1.pkl")

  return ViTDet

def copy_rpn(kp_vitdet, vitdet):
  state_dict = kp_vitdet.rpn.state_dict()

  state_dict["head.conv.0.0.weight"] = vitdet.proposal_generator.state_dict()["rpn_head.conv.conv0.weight"]
  state_dict["head.conv.0.0.bias"]   = vitdet.proposal_generator.state_dict()["rpn_head.conv.conv0.bias"]

  state_dict["head.conv.1.0.weight"] = vitdet.proposal_generator.state_dict()["rpn_head.conv.conv1.weight"]
  state_dict["head.conv.0.0.bias"]   = vitdet.proposal_generator.state_dict()["rpn_head.conv.conv1.bias"]


  state_dict["head.cls_logits.weight"] = vitdet.proposal_generator.state_dict()["rpn_head.objectness_logits.weight"]
  state_dict["head.cls_logits.bias"]   = vitdet.proposal_generator.state_dict()["rpn_head.objectness_logits.bias"]

  state_dict["head.bbox_pred.weight"] = vitdet.proposal_generator.state_dict()["rpn_head.anchor_deltas.weight"]
  state_dict["head.bbox_pred.bias"]   = vitdet.proposal_generator.state_dict()["rpn_head.anchor_deltas.bias"]

  kp_vitdet.rpn.load_state_dict(state_dict)

def vit_base_patch16(img_size, **kwargs):
  model = ViT(img_size=img_size, **kwargs)
  return model

class ViT_backbone(nn.Module):
  def __init__(self):
    super(ViT_backbone, self).__init__()

    self.img_size = 1024
    self.patch_size = 16
    self.embed_dim = 768

    ViT = vit_base_patch16(
      img_size=self.img_size,
      patch_size=self.patch_size,
      embed_dim=self.embed_dim,
      use_rel_pos=True
    )
    #ViT.load_state_dict(th.load("mae_pretrain_vit_base.pth")["model"], strict=False)
    ViT.out_channels = self.embed_dim
    self.core = SimpleFeaturePyramid(
      net=ViT,
      in_feature="last_feat",
      out_channels=256,
      scale_factors=[4, 2, 1, 1/2],
      norm="LN",
      top_block=LastLevelMaxPool(),
    )

    self.out_channels = 256
    self.out_features = self.core._out_features

  
  def forward(self, x):
    assert x.shape[-2:] == (self.img_size, self.img_size),\
      f"Input image size must be {self.img_size}x{self.img_size}, but got {x.shape[-2:]}"

    return self.core(x)

def create_KetpointRCNN(backbone, num_classes, **kwargs):

  box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=backbone.out_features,
    output_size=7,
    sampling_ratio=2
  )

  keypoint_roi_pool = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=backbone.out_features,
    output_size=14,
    sampling_ratio=2
  )

  box_predictor = FastRCNNPredictor(
    1024,
    num_classes
  )

  model = KeypointRCNN(
    backbone=backbone,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
    box_roi_pool=box_roi_pool,
    box_predictor=box_predictor,
    keypoint_roi_pool=keypoint_roi_pool,
    **kwargs
  )
  
  return model

def ViTDet(num_classes=2, pre_trained=True):
  def conv_block():
      return torchvision.ops.Conv2dNormActivation(
        256, 256, 3, stride=1, padding=1, norm_layer=LayerNorm, activation_layer=nn.ReLU
      )
    
  class BoxHead(nn.Module):
    def __init__(self):
      super(BoxHead, self).__init__()

      self.layers = nn.Sequential(
        conv_block(),
        conv_block(),
        conv_block(),
        conv_block(),
        nn.Flatten(),
        nn.Linear(256*7*7, 1024),
      )
    
    def forward(self, x):
      return self.layers(x)

  backbone = ViT_backbone()

  rpn_head = RPNHead(backbone.out_channels, 3, conv_depth=2)

  kp_vitdet = create_KetpointRCNN(
    backbone,
    num_classes,
    rpn_head=rpn_head,
    box_head=BoxHead(),
    min_size=backbone.img_size,
    max_size=backbone.img_size,
    fixed_size=(backbone.img_size, backbone.img_size)
  )

  # Get original ViTDet weights
  original_vitdet = get_original_vitdet(pre_trained)

  # Copy ViT backbone weights
  kp_vitdet.backbone.core = deepcopy(original_vitdet.backbone)

  # Copy ViTDet Region Proposal Network weights
  copy_rpn(kp_vitdet, original_vitdet)

  return kp_vitdet