#!/bin/bash

python  SumGan.py \
  --mode "decode" \
  --coverage True \
  --beam_size 6 \
  --coverage True \
  --single_pass True \
  --max_dec_steps 1\
