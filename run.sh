#!/bin/bash

for train_data_pct in 5 25 50 80; do {
for math_operator in -; do {
for weight_decay in 0 1; do {
for dropout in 0.0 0.1; do {
for opt in adamw sgd; do {
for max_lr in 0.001 0.01; do {
for random_seed in 0 100 500; do {
. train.sh $train_data_pct $math_operator $weight_decay $dropout $opt $max_lr $random_seed
} done
} done
} done
} done
} done
} done
} done

# exp="adam_subtraction_30_wd0_do0"
# #exp="adam_subtraction_30_wd1_do0.1"
# #exp="sgd_subtraction_80_wd0_do0"

# if [ $exp = "adam_subtraction_30_wd0_do0" ]; then
#     for random_seed in 1 2 3 4; do
#         ./scripts/train.py --train_data_pct 30 --group_name $exp --use_wandb --math_operator - --weight_decay 0 --dropout 0 --random_seed $random_seed
#     done
# elif [ $exp = "adam_subtraction_30_wd1_do0.1" ]; then
#     for random_seed in 1 2 3 4; do
#         ./scripts/train.py --train_data_pct 30 --group_name $exp --use_wandb --math_operator - --weight_decay 1 --dropout 0.1 --random_seed $random_seed
#     done
# elif [ $exp = "sgd_subtraction_80_wd0_do0" ]; then
#     for random_seed in 1 2 3 4; do
#         ./scripts/train.py --train_data_pct 80 --group_name $exp --use_wandb --math_operator - --weight_decay 0 --dropout 0 --opt sgd --max_lr 0.01 --random_seed $random_seed
#     done
# fi