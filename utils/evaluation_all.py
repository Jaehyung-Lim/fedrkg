import torch
import numpy as np
from sklearn.metrics import ndcg_score, recall_score
import time

def fed_evaluate_full(config, engine, train_matrix, test_matrix):
    
    mask = train_matrix >0
    labels = test_matrix

    
    score_mat = engine.get_score_mat()
    score_mat[mask] = -torch.inf
    
    metrics = evaluate(score_mat, labels)
    
    return  metrics['Recall@5'], metrics['NDCG@5'], metrics['Recall@10'], metrics['NDCG@10'], metrics['Recall@20'], metrics['NDCG@20']
        
        
def ndcg(score_mat, rel_score):
    
    NDCG = []
    
    top_k=50
    
    _, top_k_idx = torch.topk(score_mat, top_k, dim = 1)
    sorted_rel_score, _ = torch.topk(rel_score, top_k, dim=1) 
    
    for k in [5, 10, 20]:
        k_idx = top_k_idx[:, :k]
        sorted_k_rel_score = sorted_rel_score[:, :k]
        
        batch_idx = torch.arange(rel_score.size(0)).unsqueeze(1).expand(-1, k).cuda()
        
        k_rel_score = rel_score[batch_idx, k_idx]
        
        log_positions = torch.log2(torch.arange(2, k + 2, dtype = torch.float32)).cuda()
        
        dcg_k = (k_rel_score / log_positions).sum(dim=1)
        idcg_k = (sorted_k_rel_score / log_positions).sum(dim=1)
        
        tested_user = idcg_k != 0
        
        ndcg_k = dcg_k[tested_user] / idcg_k[tested_user]
        NDCG.append(round(ndcg_k.mean().detach().cpu().item(), 4))

    return NDCG


def recall(score_mat, test_mat):
    
    tested_user = test_mat.sum(dim=1) != 0 
    y_true = test_mat[tested_user]
    
    score_mat_tested = score_mat[tested_user]
    
    y_pred = torch.zeros_like(score_mat_tested).cuda()
    
    RECALL = []
    
    for k in [5, 10, 20]:
        _, top_k_item_idx = torch.topk(score_mat_tested, k, dim=1)

        batch_idx = torch.arange(top_k_item_idx.size(0)).unsqueeze(1)
        y_pred[batch_idx, top_k_item_idx] = 1.0

        y_true_ = y_true.clone().detach().cpu().numpy()
        y_pred_ = y_pred.clone().detach().cpu().numpy()

        tp = (y_true_ * y_pred_).sum(axis=1) 
        tp_fn = y_true_.sum(axis=1)  
        
        RECALL_K = (tp/tp_fn).mean().item()
        
        RECALL.append(round(RECALL_K, 4))
    
    return RECALL
    

def evaluate(score_mat, test_mat_tensor):    
    score_mat = score_mat.cuda()
    test_mat_tensor = test_mat_tensor.cuda()
    
    NDCG = ndcg(score_mat, test_mat_tensor) # N@10, N@20
    RECALL = recall(score_mat, test_mat_tensor) # R@10, R@20
    
    NR = NDCG + RECALL 
    # head = ["N@5", "N@10", "N@20", "N@50", "R@5", "R@10", "R@20", "R@50"]
    head = ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@5", "Recall@10", "Recall@20"]
    # head = ["N@20", "R@20"]
    RESULT = {}
    
    for i, metric in zip(NR, head):
        RESULT[metric] = i

    return RESULT
