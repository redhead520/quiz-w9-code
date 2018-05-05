#! /bin/bash
python train.py \
--checkpoint_path data/vgg_16.ckpt \
--output_dir ./output \
--dataset_train data/fcn_train.record \
--dataset_val data/fcn_val.record \
--batch_size 16 \
--max_steps 2000 \
--upsample_factor 8