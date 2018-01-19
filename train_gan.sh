#!/bin/bash

python  SumGan.py \
  --mode "train_gan" \
  --batch_size 4 \
  --num_models 3 \
  --single_pass False \
  --sample_num 4 \
  --gan_lr 0.00001 \
  --dis_vocab_file vocab \
