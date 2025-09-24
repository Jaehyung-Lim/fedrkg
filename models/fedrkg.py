import sys
import os
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
from utils.util import *
from tensorboardX import SummaryWriter

from utils.evaluation_all import *
import random
import copy
from utils.data import UserItemRatingDataset
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace
import torch.nn.functional as F
import math

def global_weight_finder(weight_name, config):
    if weight_name.startswith('ffn'):
        if config['ffn_locality'] == 'global':
            return True
        
        else:
            return False

    else:
        return False

# Personalized Federated Recommendation with Intermittent Guidance
class FedRKG(torch.nn.Module):
    def __init__(self, config):
        super(FedRKG, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.use_u = (config['use_u'] == 'use_u')
        
        self.sigmoid = torch.nn.Sigmoid()
        self.user_emb = None
        
        self.item_emb = torch.nn.Embedding(self.num_items, self.latent_dim)
        
        # without user embedding -> only item embedding
        self.ffn = torch.nn.Linear(self.latent_dim, 1)
        
        if self.use_u:
            self.user_emb = torch.nn.Embedding(1, self.latent_dim)
            self.ffn = torch.nn.Linear(2 * self.latent_dim, 1)
        
        self.gate_input_dim = self.latent_dim

        if self.use_u:
            if config['gate_input'] == 'u:l:g':
                self.gate_input_dim = 3 * self.latent_dim
            elif config['gate_input'] == 'u:l-g':
                self.gate_input_dim = 2 * self.latent_dim
            elif config['gate_input'] == 'l:g:l-g':
                self.gate_input_dim = 3 * self.latent_dim
            elif config['gate_input'] == 'l:g':
                self.gate_input_dim = 2 * self.latent_dim
            elif config['gate_input'] == 'l-g':
                self.gate_input_dim = self.latent_dim

        else:
            if config['gate_input'] == 'l:g:l-g':
                self.gate_input_dim = 3 * self.latent_dim
            elif config['gate_input'] == 'l:g':
                self.gate_input_dim = 2 * self.latent_dim
            elif config['gate_input'] == 'l-g':
                self.gate_input_dim = self.latent_dim
        
        self.gate = torch.nn.Linear(self.gate_input_dim, 1, bias=True)

        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)


    def forward(self, item_indices):
        if self.use_u:
            user_emb = self.user_emb(torch.LongTensor([0 for i in range(len(item_indices))]).cuda())
            item_emb = self.item_emb(item_indices)
            vector = torch.cat([user_emb, item_emb], dim=-1)  # the concat latent vector
        else:
            vector = self.item_emb(item_indices)
            
        out = self.ffn(vector)

        return out

    def forward_test(self, item_indices):
        if self.use_u:
            user_emb = self.user_emb(torch.LongTensor([0 for i in range(len(item_indices))]).cuda())
            item_emb = self.item_emb(item_indices)
            vector = torch.cat([user_emb, item_emb], dim=-1)  # the concat latent vector
        else:
            vector = self.item_emb(item_indices)
            
        out = self.ffn(vector)

        return out

    def forward_with_gate(self, item_indices, SIE):
        # SIE shape: [num_item, dimension]
        gate_input = None
        vector=None

        if self.use_u:
            user_emb = self.user_emb(torch.LongTensor([0 for i in range(len(item_indices))]).cuda())
            item_emb = self.item_emb(item_indices)
            
            gate_input = torch.cat([item_emb, SIE[item_indices], item_emb - SIE[item_indices]], dim=-1)
        
        else:
            item_emb = self.item_emb(item_indices)
            gate_input = torch.cat([item_emb, SIE[item_indices], item_emb - SIE[item_indices]], dim=-1)
            
        gate_logit = self.gate(gate_input)

        gate_weight=self.sigmoid(gate_logit)

        input_item_emb = None
        
        # ex) 0.99, 0.999, ...
        input_item_emb = item_emb * self.config['alpha'] + SIE[item_indices] * gate_weight * (1 - self.config['alpha']) * 2
        

        if self.use_u:
            vector = torch.cat([user_emb, input_item_emb], dim=-1)
        else:
            vector = input_item_emb

        out = self.ffn(vector)
        return out
    
    def gate_inference(self, item_indices, SIE):
        
        gate_input = None
        
        item_emb = self.item_emb(item_indices)
        gate_input = torch.cat([item_emb, SIE[item_indices], item_emb - SIE[item_indices]], dim=-1)
        
        gate_logit = self.gate(gate_input)
        gate_weight = self.sigmoid(gate_logit)
        
        return gate_weight.detach().clone()
            

        
class FedRKG_Engine(torch.nn.Module):
    def __init__(self, config):
        super(FedRKG_Engine, self).__init__()
        self.config = config
        self.server_model_param = {}
        self.client_model_params = {}
        self.ex_int = config['ex_int']
        self.num_round = config['num_round']
        self.global_item_emb = None
        self.global_gate = None

        self.before_guidance = {}
        
        self.use_u = (config['use_u'] == 'use_u')

        self.crit = torch.nn.BCEWithLogitsLoss()
        
        self.model = FedRKG(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        
        self.ex_rounds = [i for i in range(self.ex_int, self.num_round, self.ex_int)]

        self.lr_network = config['lr']
        self.lr_u = config['lr']
        self.lr_i = config['lr'] * config['num_items'] * config['lr_eta']
        self.lr_g = config['lr_g']

    
    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizer, user, global_epoch):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            
            
        optimizer.zero_grad()
        ratings_pred = model_client(items)
        
        loss = self.crit(ratings_pred.view(-1), ratings) 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_client.parameters(), max_norm=1.0)
        optimizer.step()
        
        return model_client, loss.item()

    def fed_train_gate_single_batch(self, model_client, batch_data, optimizers):

        optimizer, optimizer_freeze = optimizers

        """train a batch for gating unit and return an updated model."""
        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            
        optimizer.zero_grad()
        optimizer_freeze.zero_grad()
        ratings_pred = model_client.forward_with_gate(items, self.global_item_emb.data.cuda())
        
        loss = self.crit(ratings_pred.view(-1), ratings) 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_client.parameters(), max_norm=1.0)
        optimizer.step()
        
        return model_client, loss.item()
        

    def aggregate_clients_params(self, client_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.

        t = 0
        for user in client_params.keys():
            # load a user's parameters.
            user_params = client_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1

        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(client_params)

    def aggregate_item_emb(self, client_item_emb):
        t = 0
        for user in client_item_emb.keys():
            item_emb = client_item_emb[user]

            if t == 0:
                self.global_item_emb = copy.deepcopy(item_emb)
            
            else:
                self.global_item_emb.data += item_emb.data
            t+=1

        self.global_item_emb.data = self.global_item_emb.data / len(client_item_emb)

    def fed_train_gate(self, all_train_data, round_id):
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
        
        round_participant_params = {}
        client_item_emb = {}
        losses = {}

        # =========================== Nested Training Process ===========================
        for user in range(self.config['num_users']):
            loss = 0
            model_client = copy.deepcopy(self.model)

            user_param_dict = copy.deepcopy(self.model.state_dict())
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            for key in self.server_model_param.keys():
                if global_weight_finder(key, self.config):
                    user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).cuda()

            model_client.load_state_dict(user_param_dict)

            optimizer_g = torch.optim.SGD([{'params': model_client.gate.parameters(), 'lr': self.lr_g}])
            optimizer_freeze = None
            
            # we do not update other parameters during local gate training

            if self.use_u:
                optimizer_freeze = torch.optim.SGD([
                    {'params': model_client.ffn.parameters(), 'lr': 0},
                    {'params': model_client.user_emb.parameters(), 'lr': 0},
                    {'params': model_client.item_emb.parameters(), 'lr': 0},
                    ], weight_decay=self.config['l2_regularization'])
            else:
                optimizer_freeze = torch.optim.SGD([
                    {'params': model_client.ffn.parameters(), 'lr': 0},
                    {'params': model_client.item_emb.parameters(), 'lr': 0},
                    ], weight_decay=self.config['l2_regularization'])
            
            optimizer = [optimizer_g, optimizer_freeze]

            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            
            loss = 0

            sample_num = 0
            client_losses = []

            for epoch in range(self.config['gate_epoch']):
                for batch in user_dataloader:
                    model_client, client_loss = self.fed_train_gate_single_batch(model_client, batch, optimizer)

                    loss += client_loss * len(batch[0])
                    sample_num += len(batch[0])
                losses[user] = loss / sample_num
                client_losses.append(loss/sample_num)
            
                if epoch > 0 and abs(client_losses[epoch] - client_losses[epoch - 1]) / abs(
                        client_losses[epoch - 1]+ 1e-8) < self.config['tol']:
                    break
            
            client_param = model_client.state_dict()
            
            round_participant_params[user] = {}

            for key in client_param.keys():
                if key.startswith('gate'):
                    self.client_model_params[user][key].data = client_param[key].data.cpu()
            

        w_list = []

        for user in range(self.config['num_users']):

            model_client = copy.deepcopy(self.model)

            user_param_dict = copy.deepcopy(self.model.state_dict())

            for key in self.client_model_params[user].keys():
                user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()   
            
            model_client.load_state_dict(user_param_dict)

            # =========================== get gate weight ===========================
            gate_weight = model_client.gate_inference(torch.tensor(torch.arange(self.config['num_items'])).cuda(), self.global_item_emb.data)

            # =========================== knowledge fusion ===========================
            w = gate_weight.cpu()
            w_list.append(w)

            alpha = self.config['alpha']
            self.client_model_params[user]['item_emb.weight'].data = self.client_model_params[user]['item_emb.weight'].data * alpha + \
                                                                        self.global_item_emb.data.cpu() * (1- alpha) * 2 * w
            
        

    def fed_train_a_round(self, all_train_data, round_id):
        """train a round."""
        if round_id - 1 in self.ex_rounds:
            self.fed_train_gate(all_train_data, round_id)


        # sample users participating in single round.
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
        
        round_participant_params = {}
        client_item_emb = {}
        losses = {}
        for user in participants:
            loss = 0
            model_client = copy.deepcopy(self.model)
            
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                for key in self.server_model_param.keys():
                    if global_weight_finder(key, self.config):
                        user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data).cuda()

                model_client.load_state_dict(user_param_dict)
            
            
            optimizer = None
            if self.use_u:
                optimizer = torch.optim.SGD([
                    {'params': model_client.ffn.parameters(), 'lr': self.lr_network},
                    {'params': model_client.user_emb.parameters(), 'lr': self.lr_u},
                    {'params': model_client.item_emb.parameters(), 'lr': self.lr_i},
                    ], weight_decay=self.config['l2_regularization'])
            else:
                optimizer = torch.optim.SGD([
                    {'params': model_client.ffn.parameters(), 'lr': self.lr_network},
                    {'params': model_client.item_emb.parameters(), 'lr': self.lr_i},
                    ], weight_decay=self.config['l2_regularization'])
            
                
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            
            sample_num = 0
            client_losses = []
            
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, client_loss = self.fed_train_single_batch(model_client, batch, optimizer, user, round_id)
                    loss += client_loss * len(batch[0])
                    sample_num += len(batch[0])
                losses[user] = loss / sample_num
                client_losses.append(loss / sample_num)
                
                # check convergence
                if epoch > 0 and abs(client_losses[epoch] - client_losses[epoch - 1]) / abs(
                        client_losses[epoch - 1]+ 1e-8) < self.config['tol']:
                    break
            
            client_param = model_client.state_dict()
            
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            
            round_participant_params[user] = {}
            for key in self.client_model_params[user].keys():
                    if global_weight_finder(key, self.config):
                        round_participant_params[user][key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            
            client_item_emb[user] = {}
            if round_id in self.ex_rounds:
                client_item_emb[user] = copy.deepcopy(self.client_model_params[user]['item_emb.weight'].data).cuda()
                
        self.aggregate_clients_params(round_participant_params)

        if round_id in self.ex_rounds:
            self.aggregate_item_emb(client_item_emb)
        
        self.lr_network = self.lr_network * self.config['decay_rate']
        self.lr_i = self.lr_i * self.config['decay_rate']
        self.lr_u = self.lr_u * self.config['decay_rate']
        
        return participants

    def fed_evaluate(self, train_mask, test_label):
        
        mask = train_mask > 0
        
        score_mat = self.get_score_mat()
        score_mat[mask] = -torch.inf
        
        metrics = evaluate(score_mat, test_label)
            
        return metrics['Recall@'], metrics['NDCG@5'], metrics['Recall@10'], metrics['NDCG@10'], metrics['Recall@20'], metrics['NDCG@20']

    def get_score_mat(self):
        score_mat = torch.zeros(self.config['num_users'], self.config['num_items'])
        
        for user in range(self.config['num_users']):
            user_model = copy.deepcopy(self.model)
            if user in self.client_model_params.keys():
                user_param_dict = copy.deepcopy(self.client_model_params[user])
                for key in user_param_dict.keys():
                    user_param_dict[key] = user_param_dict[key].data.cuda() # user personalized score function
            for key in self.server_model_param.keys():
                    if global_weight_finder(key, self.config):
                        user_param_dict[key] = copy.deepcopy(self.server_model_param[key].data)

            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                user_score = user_model.forward_test(torch.arange(self.config['num_items']).cuda())
            score_mat[user] = user_score.cpu().reshape(-1)
            
        return score_mat