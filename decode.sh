#!/bin/bash

python  SumGan.py \
  --mode decode \
  --batch_size 14 \
  --beam_size 6 \
  --batch_size 10 \
  --single_pass True \
  # --decode_from_gan True \
  # --dec_dir []
