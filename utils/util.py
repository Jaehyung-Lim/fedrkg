"""
    Some handy functions for pytroch model training ...
"""
import torch
import numpy as np
import copy
from sklearn.metrics import pairwise_distances
import logging
import math
import os

import torch.backends.cudnn as cudnn
import random
import csv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
MODEL_ROOT = os.path.join(PROJECT_ROOT, 'models')
RESULT_ROOT = os.path.join(PROJECT_ROOT, 'fedrec_result')



# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_result_as_csv(args, RESULT):

    hyper_param = get_hyper_param(args)

    dir_path = os.path.join(RESULT_ROOT, args.dataset)
    os.makedirs(dir_path, exist_ok=True)

    title = args.model
        
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    
    csv_path = os.path.join(dir_path, title + ".csv")
    
    file_exists = os.path.isfile(csv_path)
    
    csv_head = ["seed", "N@5", "R@5", "N@10", "R@10", "N@20", "R@20", "Best_epoch"] + list(hyper_param.keys())
    row = [args.seed] + [RESULT['N@5'], RESULT['R@5'],RESULT['N@10'], RESULT['R@10'],RESULT['N@20'], RESULT['R@20'], RESULT["Best_epoch"]] + list(hyper_param.values())
    
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        if not file_exists:
            writer.writerow(csv_head)  # Write header if the file does not exist
        writer.writerow(row)  # Write the data row
        
def get_hyper_param(args):    
    hyper_param = {"job_id": args.job_id, "seed": args.seed,"dim": args.latent_dim ,"lr": args.lr, "lr_eta": args.lr_eta, 'local_epoch': args.local_epoch, 'max_round': args.num_round, 'weight_decay': args.l2_regularization, "decay_rate": args.decay_rate}

    if args.model == 'fedrig':
        hyper_param.update({'ex_int': args.ex_int, 'use_u': args.use_u, 'ffn_locality': args.ffn_locality,
                            'lr_g': args.lr_g, 'gate_epoch': args.gate_epoch, 'gate_locality': args.gate_locality, 
                            'gate_input': args.gate_input, 'alpha': args.alpha})

    elif args.model == 'pfedrig':
        hyper_param.update({'ex_int': args.ex_int,
                            'lr_g': args.lr_g, 'gate_epoch': args.gate_epoch, 'gate_locality': args.gate_locality, 
                            'gate_input': args.gate_input, 'alpha': args.alpha})


    return hyper_param

def is_bceloss_range_error(e: Exception) -> bool:
    msg = str(e)
    return (
        isinstance(e, RuntimeError)
        and "device-side assert triggered" in msg
        and "`input_val >= zero && input_val <= one` failed" in msg
    )
