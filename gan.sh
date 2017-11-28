#!/bin/bash

python  SumGan.py \
  --mode "gan" \
  --batch_size 4 \
  --num_models 3 \
  --single_pass False \
  --max_dec_steps 1\
  --beam_size 4 \
  # --coverage True \
