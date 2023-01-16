#
# Created in 2022 by Gaëtan Serré
#

import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from vision.coco_utils import CocoDetection
from vision.engine import train_one_epoch, evaluate
import vision.utils as utils
import vision.transforms as T
from vitdet import ViTDet
from vision.coco_utils import get_coco_kp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(42)

def cli():
    parser = argparse.ArgumentParser(description="Train human pose estimation ViTDet")
    parser.add_argument("--epochs", "-e", default=10, type=int,
                        help="Number of epochs to train (default: 10)")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size (default: 4)")             
    parser.add_argument("--output_path", "-o", default="weights/", type=str,
                        help="Path to save the model (default: weights/)")
    parser.add_argument("--pretrain", action=argparse.BooleanOptionalAction,
                        default=True, help="Use pre-trained weights")
    parser.add_argument("--lr", default=5e-4, type=float,
                        help="Learning rate (default 5e-4)")

    return parser.parse_args()

if __name__ == "__main__":
  args = cli()

  if not args.pretrain:
    print("NO PRETRAINED WEIGHTS")


  # load the COCO dataset
  dataset_train = get_coco_kp("coco", "train", transforms=T.ToTensor())
  dataset_val = get_coco_kp("coco", "val", transforms=T.ToTensor())

  print(f"{len(dataset_train)} train images & {len(dataset_val)} val images.")

  # define training and validation data loaders
  data_loader_train = torch.utils.data.DataLoader(
      dataset_train, batch_size=args.batch_size, shuffle=True,
      collate_fn=utils.collate_fn)

  data_loader_val = torch.utils.data.DataLoader(
      dataset_val, batch_size=args.batch_size, shuffle=False,
      collate_fn=utils.collate_fn)
  
  num_epochs = args.epochs
  
  # Iterate through the models to test
  model = ViTDet(pre_trained=args.pretrain).to(device)
  model_name = "ViTDet"

  # construct an optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

  # let's train it
  scaler = torch.cuda.amp.GradScaler()
  print(f"\033[94mTraining {model_name} for {num_epochs} epochs.\033[0m")
  mAPs_box  = []
  mAPs_kpts = []
  for epoch in range(num_epochs):
      # train for one epoch, printing every 10 iterations
      train_one_epoch(
        model,
        optimizer,
        data_loader_train,
        device,
        epoch,
        print_freq=10,
        scaler=scaler
      )
      # update the learning rate
      scheduler.step()
      # evaluate on the test dataset
      evaluator = evaluate(
        model,
        data_loader_val,
        device,
        iou_types=["bbox", "keypoints"],
      )
      
      mAPs_box.append(evaluator.coco_eval["bbox"].stats[0])
      mAPs_kpts.append(evaluator.coco_eval["keypoints"].stats[0])

  if args.pretrain:
    save_path = os.path.join(args.output_path, f"ViTDet_{num_epochs}epochs.pth")
  else:
    save_path = os.path.join(args.output_path, f"ViTDet_noweights_{num_epochs}epochs.pth")

  torch.save(model.state_dict(), save_path)
  print(f"\033[94mModel saved to {save_path}.\033[0m")
  
  # Plot mAPs
  plt.style.use("seaborn-v0_8")
  plt.figure(figsize=(10, 5))

  plt.plot(range(1, num_epochs+1), mAPs_box, ".-", label=f"{model_name}")
  plt.grid(True)
  plt.ylabel("$mAP^{box}@[0.5:0.95]$")
  plt.xlabel("Epoch")
  plt.legend()
  if args.pretrain:
    plt.savefig("figures/mAPs_box.pdf")
    np.save("figures/bbox", mAPs_box)
  else:
    plt.savefig("figures/mAPs_noweights_box.pdf")
    np.save("figures/bbox_noweights", mAPs_box)

  plt.clf()

  plt.plot(range(1, num_epochs+1), mAPs_kpts, ".-", label=f"{model_name}")
  plt.grid(True)
  plt.ylabel("$mAP^{kpts}@[0.5:0.95]$")
  plt.xlabel("Epoch")
  plt.legend()
  if args.pretrain:
    plt.savefig("figures/mAPs_kpts.pdf")
    np.save("figures/kpts", mAPs_kpts)
  else:
    plt.savefig("figures/mAPs_noweights_kpts.pdf")
    np.save("figures/kpts_noweights", mAPs_kpts)

  print("\033[92mThat's it!\033[0m")