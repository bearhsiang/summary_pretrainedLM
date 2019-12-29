#!usr/bin/env bash
train_file=/hdd/gpt2_sum/data/train.txt
eval_file=/hdd/gpt2_sum/data/test.txt
output_dir=/hdd/gpt2_sum/model_sum
batch_size=4

python run_lm_finetuning.py \
    --train_data_file=$train_file \
    --output_dir=$output_dir \
    --eval_data_file=$eval_file \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size=$batch_size \
    --per_gpu_eval_batch_size=$batch_size \
    --overwrite_output_dir \
    --save_steps=1000 \
    --block_size=300