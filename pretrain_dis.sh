#!/bin/bash
 
python  SumGan.py \
  --mode pretrain_dis \
  --max_steps 20000000 \
  --dis_vocab vocab \
  --num_models 3 \
