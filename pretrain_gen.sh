#!/bin/bash

python  SumGan.py \
  --mode pretrain_gen \
  --decoder conv_decoder \
  --encoder conv_encoder \
  # --restore_best_model True \
