#!/bin/bash

python  SumGan.py \
  --mode "gan" \
  --batch_size 10 \
  --single_pass False \
  --max_dec_steps 1\
  --beam_size 6 \
  # --coverage True \
