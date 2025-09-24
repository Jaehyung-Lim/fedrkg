import pandas as pd
import numpy as np
import datetime
import time
import os
import torch.backends.cudnn as cudnn
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import pandas as pd
pd.options.mode.chained_assignment = None

from utils.early_stop import *
from utils.util import *

from models.fedrkg import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
parser.add_argument('--clients_sample_num', type=int, default=0)
parser.add_argument('--num_round', type=int, default=10)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--epoch_setting', type=int, default=-1)
parser.add_argument('--ffn_locality', type=str, default='local', choices=['local', 'global'])
parser.add_argument('--gate_epoch', type=int, default=5, help='gate epoch')
parser.add_argument('--ex_int', type=int, default=2, help='exchange interval')
parser.add_argument('--alpha', type=float, default=0.99, help='we use this parameter as beta in the paper.')
parser.add_argument('--use_u', type=str, default='not', choices=['use_u', 'not'])
parser.add_argument('--lr_g', type=float, default=0.1, help='learning rate for gate')
parser.add_argument('--model', type=str, default='fedrkg')
parser.add_argument('--init_param', type=int, default=1, help="0; no, 1: yes")
parser.add_argument('--gate_input', type=str, default='l:g:l-g', choices=['l:g', 'l+g', 'l-g', 'l:g:l-g', 'l:g:abs', 'u:l:g', 'u:l-g'])
parser.add_argument('--decay_rate', type=float, default=1.0)
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--lr_eta', type=int, default=80)
parser.add_argument('--batch_size', type=int, default=256) 
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--dataset', type=str, default='filmtrust', choices=['ml-1m', 'lastfm-2k', 'filmtrust', 'video'])
parser.add_argument('--num_users', type=int)
parser.add_argument('--num_items', type=int)
parser.add_argument('--latent_dim', type=int, default=32)
parser.add_argument('--num_negative', type=int, default=4)
parser.add_argument('--l2_regularization', type=float, default=0)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--job_id', type=str, default='1111')
parser.add_argument('--save_result', type=int, default=1, help='0: not save, 1: save')
args = parser.parse_args()

    
if args.epoch_setting == 1:
    args.num_round = 1000

elif args.epoch_setting == 2:
    args.num_round = 3000
    

print(args, flush=True)


# Set Seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(args.seed)

early = EarlyStopping(patience=args.patience, larger_is_better = True) 

# Model.
config = vars(args)

if config['dataset'] == 'ml-1m': 
    config['num_users'] = 6040
    config['num_items'] = 3706
elif config['dataset'] == 'lastfm-2k': 
    config['num_users'] = 1269
    config['num_items'] = 12322
elif config['dataset'] == 'video': 
    config['num_users'] = 1372
    config['num_items'] = 7957
elif config['dataset'] == 'filmtrust':
    config['num_users'] = 1227
    config['num_items'] = 2059
else:
    pass


# Define FL Engine
engine = FedRKG_Engine(config)


    
# Load Data

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(PROJECT_ROOT, 'data', config['dataset'], 'ratings.dat')

if config['dataset'] == "ml-1m":
    rating = pd.read_csv(dataset_dir, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
elif config['dataset'] == "lastfm-2k":
    rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
elif config['dataset'] == 'video':
    rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
elif config['dataset'] == 'filmtrust':
    rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
else:
    pass

# Reindex
user_id = rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
rating = pd.merge(rating, user_id, on=['uid'], how='left')
item_id = rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
rating = pd.merge(rating, item_id, on=['mid'], how='left')
rating = rating[['userId', 'itemId', 'rating', 'timestamp']]

# DataLoader for training

from utils.data import SampleGenerator
sample_generator = SampleGenerator(ratings=rating)
train_mat = sample_generator.train_mat
valid_mat = sample_generator.valid_mat
test_mat = sample_generator.test_mat

    

hit_ratio_5_list = []
ndcg_5_list = []
hit_ratio_10_list = []
ndcg_10_list = []
hit_ratio_20_list = []
ndcg_20_list = []
train_loss_list = []
test_loss_list = []
val_loss_list = []
best_val_hr = 0
final_test_round = 0
best_client_param = {}
best_server_param = {}
best_score_mat = 0


start = time.time()

try:
    for rnd in range(config['num_round']):
    
        start_rnd = time.time()
        print('-' * 120,flush=True)
        print(f"Round {rnd} starts !.",flush=True)

        all_train_data = sample_generator.store_all_train_data(config['num_negative'])
        tr_loss = engine.fed_train_a_round(all_train_data, round_id=rnd)
        train_loss_list.append(tr_loss)
        
        
        hr_5, ndcg_5, hr_10, ndcg_10, hr_20, ndcg_20 = fed_evaluate_full(config, engine, train_mat + valid_mat, test_mat)
        
        
        print('[Testing Epoch {}] NDCG@5 = {:.4f}, HR@5 = {:.4f}, NDCG@10 = {:.4f}, HR@10 = {:.4f}, NDCG@20 =  {:.4f}, HR@20 = {:.4f}'.format(rnd, ndcg_5, hr_5, ndcg_10, hr_10, ndcg_20, hr_20),flush=True)
        hit_ratio_5_list.append(hr_5)
        ndcg_5_list.append(ndcg_5)
        hit_ratio_10_list.append(hr_10)
        ndcg_10_list.append(ndcg_10)
        hit_ratio_20_list.append(hr_20)
        ndcg_20_list.append(ndcg_20)
        val_hr_5, val_ndcg_5, val_hr_10, val_ndcg_10, val_hr_20, val_ndcg_20 = fed_evaluate_full(config, engine, train_mat, valid_mat)
        
        
            
        end_rnd = time.time()
        print('[Evluating Epoch {}] NDCG@5 = {:.4f}, HR@5 = {:.4f}, NDCG@10 = {:.4f}, HR@10 = {:.4f}, NDCG@20 =  {:.4f}, HR@20 = {:.4f}'.format(rnd, val_ndcg_5, val_hr_5, val_ndcg_10, val_hr_10, val_ndcg_20, val_hr_20),flush=True)
        print(f"Running 1 epoch: {(end_rnd-start_rnd)/60:.2f} min", flush=True)

        if val_hr_10 >= best_val_hr:
            best_val_hr = val_hr_10
            final_test_round = rnd
            # best_score_mat = engine.get_score_mat()
            best_client_param = copy.deepcopy(engine.client_model_params)
            best_server_param = copy.deepcopy(engine.server_model_param)
        
        early(val_hr_10)
        if early.stop_flag:
            print(f"=================== Early Stopping!!!! ===================",flush=True)
            break
        
        
except Exception as e:
    if is_bceloss_range_error(e):
        print(f"\n [BCELoss Range Error] Ignoring known device-side assert in round {rnd}\n", flush=True)
    else:
        print(f"\n [Unhandled Error] Training failed at round {rnd}: {e}\n", flush=True)
        import traceback
        traceback.print_exc()
    
finally:
    print('Best test ndcg@5: {}, hr@5: {} ndcg@10: {}, hr@10: {} ndcg@20: {}, hr@20: {} at round {}'.format(ndcg_5_list[final_test_round],
                                                                                                            hit_ratio_5_list[final_test_round],
                                                                                                            ndcg_10_list[final_test_round],
                                                                                                            hit_ratio_10_list[final_test_round],
                                                                                                            ndcg_20_list[final_test_round],
                                                                                                            hit_ratio_20_list[final_test_round],
                                                                                                            final_test_round),flush=True)


    end = time.time()
    run_time = (end - start)/50
    print(f"Running time: {(end - start)/60:.2f} min", flush=True)

    

    if config['save_result'] == 1:
        save_result_as_csv(args, {  "N@5": round(ndcg_5_list[final_test_round], 4), "R@5": round(hit_ratio_5_list[final_test_round], 4),
                                    "N@10": round(ndcg_10_list[final_test_round], 4), "R@10": round(hit_ratio_10_list[final_test_round], 4),
                                    "N@20": round(ndcg_20_list[final_test_round], 4), "R@20": round(hit_ratio_20_list[final_test_round], 4),
                                    "Best_epoch": final_test_round})