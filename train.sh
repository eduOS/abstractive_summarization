#!/bin/bash
export CUDA_VISIBLE_DEVICES="8,9,10,11"
      python  tfGan.py --is_decode False \
        --is_generator_train False \
				--is_gan_train True \
				--is_discriminator_train False \
        --train_data_source '~/data/LCSTS/finished_files/train-art.txt' \
        --train_data_target '~/data/LCSTS/finished_files/train-abs.txt' \
        --clip_c 10.0  \
        --batch_size 64  \
        --max_len 50 \
        --saveto 'nmt180w_second'  \
        --dis_saveto 'disCnn-180w-multi-generator' \
        --dis_max_len 50 \
        --gpu_device 'gpu-0' \
        --dis_gpu_device 'gpu-0' \
        --dis_batch_size 40 \
        --gan_gen_batch_size 100 \
        --dis_dispFreq 1 \
        --dis_epoches 1 \
        --gen_reload True \
        --dis_reload True \
        --teacher_forcing True \
        --gan_total_iter_num 500 \
        --gan_gen_iter_num 1 \
        --gan_dis_iter_num 1 \
        --gan_dispFreq 10 \
        --gan_saveFreq 500 \
        --roll_num 20 \
        --generate_num 5000 \
        --bias_num 0.5 \
        # --gan_gen_source_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/gan_gen_source_u8.txt' \
        # --gan_dis_negative_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/gan_dis_negative_u8.txt' \
        --gan_dis_negative_data './data_for_multi_gan_train/gan_dis_negative.txt' \
        # --gan_dis_positive_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/gan_dis_positive_u8.txt' \
        --gan_dis_positive_data './data_for_multi_gan_train/gan_dis_positive.txt' \
        # --dis_dev_negative_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/dev_negative_u8.txt' \
        --dis_dev_negative_data './data_for_multi_gan_train/dev_negative.txt' \
        # --dis_dev_positive_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/dev_positive_u8.txt' \
        --dis_dev_positive_data './data_for_multi_gan_train/dev_positive_u8.txt' \
        # --dis_dev_source_data '/data/zhyang/dl4mt/corpus/data_gan_180w_zxw/data_for_multi_gan_train/dev_source_u8.txt'
        --dis_dev_source_data './data_for_multi_gan_train/dev_source.txt'
