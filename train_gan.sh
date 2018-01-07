#!/bin/bash

python  SumGan.py \
  --mode "train_gan" \
  --batch_size 4 \
  --num_models 3 \
  --single_pass False \
  --beam_size 4 \
  --gan_lr 0.00001
  # --coverage True \
