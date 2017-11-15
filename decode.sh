#!/bin/bash

python  SumGan.py \
  --mode "decode" \
  --beam_size 6 \
  --single_pass True \
  --max_dec_steps 1
