#!/bin/bash

python  SumGan.py \
  --mode decode \
  --beam_size 6 \
  --batch_size 10 \
  --single_pass True \
  # --dec_dir []
