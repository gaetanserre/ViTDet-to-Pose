#!/bin/sh

set -x

python train_vitdet.py \
--epochs 10 \
--batch_size 4 \
--output_path weights \
--lr 1e-4 \
