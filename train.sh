#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
python  tfGan.py --is_decode False \
  --is_generator_train True \
  --is_gan_train False \
  --is_discriminator_train False \
  --dict_path '/home/lerner/data/LCSTS/finished_files/vocab' \
  --train_data_source '/home/lerner/data/LCSTS/finished_files/train-art.txt' \
  --train_data_target '/home/lerner/data/LCSTS/finished_files/train-abs.txt' \
  --decode_file '/home/lerner/data/LCSTS/finished_files/train-art.txt' \
  --decode_result_file './data_train/negative.txt' \
  --clip_c 10.0  \
  --batch_size 64  \
  --max_len_s 80 \
  --saveto './gen_model/lcsts'  \
  --dis_saveto './dis_model/lcsts' \
  --max_leng 15 \
  --gpu_device 'gpu-0' \
  --dis_gpu_device 'gpu-0' \
  --dis_batch_size 40 \
  --gan_gen_batch_size 100 \
  --dis_dispFreq 1 \
  --dis_epoches 1 \
  --gen_reload False \
  --dis_reload False \
  --teacher_forcing False \
  --gan_total_iter_num 500 \
  --gan_gen_iter_num 1 \
  --gan_dis_iter_num 1 \
  --gan_dispFreq 10 \
  --gan_saveFreq 500 \
  --roll_num 20 \
  --generate_num 5000 \
  --bias_num 0.5 \
  --gan_dis_negative_data './data_train/negative.txt' \
  --gan_dis_positive_data './data_train/positive.txt' \
  --gan_dis_source_data './data_train/source.txt' \
  --dis_negative_data './data_train/negative.txt' \
  --dis_positive_data './data_train/positive.txt' \
  --dis_source_data './data_train/source.txt' \
  --dis_dev_negative_data './data_val/negative.txt' \
  --dis_dev_positive_data './data_val/positive.txt' \
  --dis_dev_source_data './data_val/source.txt' 
  # --gan_gen_source_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/gan_gen_source_u8.txt' \
  # --gan_dis_negative_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/gan_dis_negative_u8.txt' \
  # --gan_dis_positive_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/gan_dis_positive_u8.txt' \
  # --dis_dev_negative_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/dev_negative_u8.txt' \
  # --dis_dev_positive_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/dev_positive_u8.txt' \
  # --dis_dev_source_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/dev_source_u8.txt'
