#
# Created in 2022 by Gaëtan Serré
#

import torch
import numpy as np

from ViTPose.mmpose.apis.inference import init_pose_model, inference_top_down_pose_model, process_mmdet_results

class ViTPose:
  def __init__(self, device="cpu"):

    self.core = init_pose_model(
      "ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_base_ochuman_256x192.py",
      checkpoint="weights/vit_weights/vitpose-b-multi-coco.pth",
      device=device
    )

  def eval(self):
    self.core.eval()
    return self
  
  def train(self):
    self.core.train()
    return self

  def __call__(self, im, label):
    return self.forward(im, label)

  @staticmethod
  def convert_results(pose_results, proposal):
    converted = {"boxes": [], "labels": proposal["labels"], "scores": proposal["scores"],
                 "keypoints": [], "keypoints_scores": []}
    for pose in pose_results:
      converted["boxes"].append(pose["bbox"])
      tmp_kpts = []
      tmp_kpts_scores = []
      for kpts in pose["keypoints"]:
        score = kpts[2] * 10
        kpts[2] = 1
        tmp_kpts.append(kpts)
        tmp_kpts_scores.append(score)
      converted["keypoints"].append(tmp_kpts)
      converted["keypoints_scores"].append(tmp_kpts_scores)
    
    converted["boxes"]            = torch.tensor(np.array(converted["boxes"]))
    converted["keypoints"]        = torch.tensor(np.array(converted["keypoints"]))
    converted["keypoints_scores"] = torch.tensor(np.array(converted["keypoints_scores"]))
    
    return converted

  def forward(self, im, label):
    pose_results, returned_outputs = inference_top_down_pose_model(
      self.core,
      im[0].permute(1, 2, 0).cpu().numpy(),
      person_results=process_mmdet_results([label[0]["boxes"].cpu().numpy()]),
      dataset="TopDownOCHumanDataset",
      format="xyxy"
    )
    return [self.convert_results(pose_results, label[0])]