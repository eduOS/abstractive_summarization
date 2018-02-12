#!/bin/bash

python  SumGan.py \
  --mode "train_gan" \
  --batch_size 8 \
  --num_models 3 \
  --dis_vocab_file vocab \
  --single_pass False \
  --beam_size 4 \
  --gan_lr 0.00005 \
  --gen_lr 0.00005 \
  --dis_lr 0.00005 \
  --vocab_type char \
  --dis_reward_ratio 1 \
  --rollout_num 12 \
  --sample_rate 0.001 \
  --gan_gen_iter 5 \
  --rouge_reward_ratio 0 \
  --dis_reward_ratio 1 \
  # --coverage True \
