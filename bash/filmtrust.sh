#!/bin/bash


python3 ../main.py --dataset filmtrust --model fedrkg \
                   --lr 0.1 --alpha 0.99 --use_u not \
                   --gate_locality global --ffn_locality local --seed 42 \
                   --epoch_setting 1 --decay_rate 1.0 --ex_int 100 \
                   --gate_epoch 5 --lr_g 1e-4 --gate_input l:g:l-g \
                   --save_result 1