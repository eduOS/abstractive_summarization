#!/bin/bash

python  SumGan.py \
  --mode "gan" \
  --coverage True \
  --beam_size 6 \
  --coverage True \
  --single_pass False \
  --max_dec_steps 1\
