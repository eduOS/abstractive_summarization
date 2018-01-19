#!/bin/bash
 
python  SumGan.py \
  --mode pretrain_dis \
  --max_steps 20000000 \
  --num_models 3 \
  --early_stop True \
  --dis_vocab_file vocab \
