#!/bin/bash
# . train.sh $train_data_pct $math_operator $weight_decay $dropout $opt $max_lr $random_seed $use_wandb $group_name
train_data_pct=${1-5}
math_operator=${2-+}
weight_decay=${3-1}
dropout=${4-0.0}
opt=${5-adamw}
max_lr=${6-0.001}
random_seed=${7-0}
use_wandb=True
group_name="wd=${weight_decay}-d=${dropout}-opt=${opt}-mlr=${max_lr}-rs=${random_seed}-mo${math_operator}"

./scripts/train.py \
		--batchsize 0 \
		--n_layers 2 \
		--n_heads 4 \
		--d_model 128 \
		--dropout $dropout \
		--weight_noise 0.0 \
		--non_linearity relu \
		--max_context_len 50 \
		--math_operator $math_operator \
		--train_data_pct $train_data_pct \
		--warmup_steps 10 \
		--anneal_lr_steps 100000 \
		--anneal_lr False \
		--max_lr 0.001 \
		--weight_decay $weight_decay \
		--weight_decay_kind to_zero \
		--noise_factor 0 \
		--save_activations False \
		--save_outputs False \
		--logdir logs \
		--datadir data \
		--save_checkpoint True \
		--use_wandb $use_wandb \
		--group_name $group_name \
		--opt $opt \
		--momentum 0.9 \
		--random_seed $random_seed \
		--max_steps 100000 \
#		--load_from_ckpt None \
#		--operand_length \
#		--max_epochs 1e9 \
