#
# Created in 2022 by Gaëtan Serré
#

import torch
import torchvision

import vision.transforms as T
from vision.coco_utils import CocoDetection
from vision.engine import evaluate
import vision.utils as utils
from vitpose import ViTPose
from vitdet import ViTDet
from vision.coco_utils import get_coco_kp

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# load the COCO validation dataset
dataset_test = get_coco_kp("coco", "val", transforms=T.ToTensor())

data_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False,
    collate_fn=utils.collate_fn)


vitdet = ViTDet().to(device)
print("Evaluating ViTDet.")
vitdet.load_state_dict(torch.load("weights/ViTDet_10epochs.pth"))
evaluate(vitdet, data_loader, device, iou_types=["keypoints"])


def get_bbox(images):
  vitdet.eval()
  outputs = vitdet(images)[0]
  outputs["keypoints"] = torch.empty_like(outputs["keypoints"])
  outputs["keypoints_scores"] = torch.empty_like(outputs["keypoints_scores"])
  return [outputs]

vitpose = ViTPose(device=device)
print("Evaluating ViTPose using ViTDet bbox proposal.")
evaluate(vitpose, data_loader, device, iou_types=["keypoints"], get_bbox=get_bbox)