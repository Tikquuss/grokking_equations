#!/usr/bin/env python

import grok
import os

"""
--batchsize 0
--n_layers 2
--n_heads 4
--d_model 128
--dropout 0.0
--weight_noise 0.0
--non_linearity relu
--max_context_len 50
--math_operator +
--operand_length 
--train_data_pct 5
--warmup_steps 10
--anneal_lr_steps 100000
--anneal_lr False
--max_lr 0.001
--weight_decay 1
--weight_decay_kind to_zero
--noise_factor 0
--save_activations False
--save_outputs False
--logdir logs 
--datadir data
--save_checkpoint True
--use_wandb True   
--group_name base
--load_from_ckpt None
--opt adamw 
--momentum 0.9
--random_seed 0
--max_epochs 1e9
--max_steps 100000
"""

parser = grok.training.add_args()
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)

#print(hparams)
print("*"*10, "\thparams\t", "*"*10)
for k, v in vars(hparams).items() :
    print(k, v)
print("*"*(10+11+10))
print(grok.training.train(hparams))
