import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)

class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        
        self.num_users = len(self.user_pool)
        self.num_items = len(self.item_pool)
        
        
        self.negatives = self._sample_negative(ratings)
        
        self.train_ratings, self.val_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)

        self.train_mat, self.valid_mat, self.test_mat = self._train_valid_test_mat()
        
        
        self.train_items = self._get_seen_items(self.train_ratings)
        self.val_items = self._get_seen_items(self.val_ratings)
        self.test_items = self._get_seen_items(self.test_ratings)
        
    
    def _train_valid_test_mat(self):
        train_mat = torch.zeros((self.num_users, self.num_items))
        valid_mat = torch.zeros((self.num_users, self.num_items))
        test_mat = torch.zeros((self.num_users, self.num_items))
        
        for row in self.train_ratings.itertuples():
            train_mat[row.userId, row.itemId] = 1.0
        
        for row in self.val_ratings.itertuples():
            valid_mat[row.userId, row.itemId] = 1.0
        
        for row in self.test_ratings.itertuples():
            test_mat[row.userId, row.itemId] = 1.0
            
        return train_mat, valid_mat, test_mat


    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        val = ratings[ratings['rank_latest'] == 2]
        train = ratings[ratings['rank_latest'] > 2]
        assert train['userId'].nunique() == test['userId'].nunique() == val['userId'].nunique()
        assert len(train) + len(test) + len(val) == len(ratings)
        return train[['userId', 'itemId', 'rating']], val[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        #interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 198))
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), 198))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def store_all_train_data(self, num_negatives):
        """store all the train data as a list including users, items and ratings. each list consists of all users'
        information, where each sub-list stores a user's positives and negatives"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(list(x),         
                                                                                                   num_negatives))  
        single_user = []
        user_item = []
        user_rating = []
        # split train_ratings into groups according to userId.
        grouped_train_ratings = train_ratings.groupby('userId')
        train_users = []
        for userId, user_train_ratings in grouped_train_ratings:
            train_users.append(userId)
            user_length = len(user_train_ratings)
            for row in user_train_ratings.itertuples():
                single_user.append(int(row.userId))
                user_item.append(int(row.itemId))
                user_rating.append(float(row.rating))
                for i in range(num_negatives):
                    single_user.append(int(row.userId))
                    user_item.append(int(row.negatives[i]))
                    user_rating.append(float(0))  # negative samples get 0 rating
            assert len(single_user) == len(user_item) == len(user_rating)
            assert (1 + num_negatives) * user_length == len(single_user)
            users.append(single_user)
            items.append(user_item)
            ratings.append(user_rating)
            single_user = []
            user_item = []
            user_rating = []
        assert len(users) == len(items) == len(ratings) == len(self.user_pool)
        assert train_users == sorted(train_users)
        return [users, items, ratings]

        
    def _get_seen_items(self, input_data):
        return input_data.groupby('userId')['itemId'].apply(set).to_dict()
    
    
    def _get_candidate_items(self, user_id, mode):
        seen_items = self.train_items.get(user_id, set()).copy()
        
        if mode == 'valid':
            seen_items |= self.val_items.get(user_id, set())
        elif mode == 'test':
            seen_items |= self.val_items.get(user_id, set())
            seen_items |= self.test_items.get(user_id, set())
        
        return list(self.item_pool - seen_items)
    
    def _generate_eval_data(self, eval_ratings, mode):
        eval_dict = {}
        
        for row in eval_ratings.itertuples():
            user_id = int(row.userId)
            positive_item = int(row.itemId)

            candidate_items = self._get_candidate_items(user_id, mode)

            # eval_dict: { user_id: [positive_item, negative_items] }
            eval_dict[user_id] = [torch.LongTensor([positive_item]), torch.LongTensor(candidate_items)]

        return eval_dict
    
    
    @property
    def validate_data(self):
        return self._generate_eval_data(self.val_ratings, mode="valid")
    
    @property
    def test_data(self):
        return self._generate_eval_data(self.test_ratings, mode="test")