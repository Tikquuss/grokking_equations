#!/bin/bash

### usage ###
# . train.sh $train_data_pct $math_operator $weight_decay $dropout $opt $max_lr $random_seed $use_wandb $group_name
# all this parameters are optional (see the default values below)

### params ###
train_data_pct=${1-5}
math_operator=${2-+}
weight_decay=${3-1}
dropout=${4-0.0}
opt=${5-adamw}
max_lr=${6-0.001}
random_seed=${7-0}

#max_steps=100000
max_steps=50000

### wandb ###
# wandb_entity is the name of the team on wandb and is optional
# wandb_project is the name of the project
use_wandb=True
#group_name="tdp=${train_data_pct}-wd=${weight_decay}-d=${dropout}-opt=${opt}-mlr=${max_lr}-rs=${random_seed}-mo${math_operator}"
# remove random_seed in group_name
group_name="tdp=${train_data_pct}-wd=${weight_decay}-d=${dropout}-opt=${opt}-mlr=${max_lr}-mo${math_operator}"
wandb_entity="grokking_ppsp"
wandb_project="grokking_operator=${math_operator}"

### Experiment dump path ###
dump_path=..
logdir=${dump_path}/logs/$group_name
datadir=${dump_path}/data/$group_name

### Early_stopping ###
early_stopping_patience=5000
patience_metric=val_accuracy
early_stopping_step_val_acc_threshold=90.0

###
./scripts/train.py \
		--batchsize -1 \
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
		--max_lr $max_lr \
		--weight_decay $weight_decay \
		--weight_decay_kind to_zero \
		--noise_factor 0 \
		--save_activations False \
		--save_outputs False \
		--logdir $logdir \
		--datadir $datadir \
		--save_checkpoint True \
		--use_wandb $use_wandb \
		--group_name $group_name \
		--wandb_entity $wandb_entity \
		--wandb_project $wandb_project \
		--opt $opt \
		--momentum 0.9 \
		--random_seed $random_seed \
		--max_steps $max_steps \
		--use_cuda True \
		--gpu -1 \
		--early_stopping_patience $early_stopping_patience \
		--patience_metric $patience_metric \
		--early_stopping_step_val_acc_threshold $early_stopping_step_val_acc_threshold \
#		--max_epochs 1e9 \
#		--load_from_ckpt None \
#		--operand_length \
