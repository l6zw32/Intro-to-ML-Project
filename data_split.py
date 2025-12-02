# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 20:29:17 2025

Some helper functions for generating training and testing data
"""

import random
import math
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd
import numpy as np

default_data = Dataset.load_builtin("ml-100k")


class UserKFold:
    """
    Splits the dataset into folds by users (not individual ratings).
    Each fold contains a subset of users; test set contains *all* ratings of those users.
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None, frac=0.3):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.frac = frac

    def split(self, dataset):
        raw = dataset.raw_ratings

        df = pd.DataFrame(raw, columns=["UserID", "MovieID", "Rating", "timestamp"])
        df = df.drop(columns="timestamp")

        # Group by user
        users = df['UserID'].unique().tolist()

        if self.shuffle:
            rnd = random.Random(self.random_state)
            rnd.shuffle(users)

        fold_size = math.ceil(len(users) / self.n_splits)

        for i in range(self.n_splits):
            test_users = set(users[i * fold_size:(i + 1) * fold_size])

            train_data = []
            test_data = []
            

            for uid in test_users:
                user_ratings = df[df['UserID'] == uid]

                revealed = user_ratings.sample(frac=self.frac, random_state=self.random_state)
                held_out = user_ratings.drop(revealed.index)
                train_data.append(revealed)
                test_data.append(held_out)
            
            # Add all other users' ratings to train
            # The cold_start users (with their actual ratings) are used as test set
            other_users = df[~df['UserID'].isin(test_users)]
            train_data.append(other_users)

            train_df = pd.concat(train_data)
            test_df = pd.concat(test_data)

            reader = Reader(rating_scale=(1, 5))
            trainset = Dataset.load_from_df(train_df[['UserID', 'MovieID', 'Rating']], reader).build_full_trainset()
            testset = Dataset.load_from_df(test_df[['UserID', 'MovieID', 'Rating']], reader).build_full_trainset().build_testset()
            
            yield trainset, testset



# Partial cold start
# Splits data to training and testing sets based on users
def cold_start_train(data=default_data, cold_start_user_portion=0.2, frac=0.3, random_seed=10701):
    trainset = data.build_full_trainset()

    # Convert the trainset to a list of tuples (user, item, rating)
    ratings = [(trainset.to_raw_uid(uid), trainset.to_raw_iid(iid), rating) for (uid, iid, rating) in trainset.all_ratings()]

    # Convert it into a pandas DataFrame
    ratings = pd.DataFrame(ratings, columns=['UserID', 'MovieID', 'Rating'])
    # Sample random users as "cold-start" users
    cold_start_users = ratings['UserID'].drop_duplicates().sample(frac=cold_start_user_portion, random_state=random_seed)

    train_data = []
    test_data = []
    

    for uid in cold_start_users:
        user_ratings = ratings[ratings['UserID'] == uid]

        revealed = user_ratings.sample(frac=frac, random_state=random_seed)
        held_out = user_ratings.drop(revealed.index)
        train_data.append(revealed)
        test_data.append(held_out)
    
    # Add all other users' ratings to train
    # The cold_start users (with their actual ratings) are used as test set
    other_users = ratings[~ratings['UserID'].isin(cold_start_users)]
    train_data.append(other_users)

    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)

    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(train_df[['UserID', 'MovieID', 'Rating']], reader).build_full_trainset()
    testset = Dataset.load_from_df(test_df[['UserID', 'MovieID', 'Rating']], reader).build_full_trainset().build_testset()
    
    return trainset, testset


# Complete cold start, or used for cross validation
# Splits the data by users
def user_split(data=default_data, test_size=0.2):
    reader = Reader()
    raw_data = data.raw_ratings
    users = list({uid for (uid, iid, rating, ts) in raw_data})
    random.shuffle(users)
    
    # Determine split point
    test_count = int(len(users) * test_size)
    test_users = set(users[:test_count])
    train_users = set(users[test_count:])

    # Split raw ratings by users
    train_raw = [r for r in raw_data if r[0] in train_users]
    test_raw  = [r for r in raw_data if r[0] in test_users]

    trainset = [(uid, iid, rating) for (uid, iid, rating, ts) in train_raw]    
    testset = [(uid, iid, rating) for (uid, iid, rating, ts) in test_raw]
    
    train_df = pd.DataFrame(trainset, columns =['UserID', 'MovieID', 'Rating'])
    test_df = pd.DataFrame(testset, columns =['UserID', 'MovieID', 'Rating'])
    
    trainset = Dataset.load_from_df(train_df[['UserID', 'MovieID', 'Rating']], reader).build_full_trainset()
    testset = Dataset.load_from_df(test_df, reader).build_full_trainset().build_testset()

    return trainset, testset


# partial cold start
# cross validate by splitting on users. Return trainset, testset, and results
# if trainset is defined, trainset is used, and testset is set to None. This renders data and test_size useless.
def cold_start_cross_validate(algo, data=default_data, trainset=None, test_size=0.2 , k=5):
    testset = None
    if trainset == None:
        trainset, testset = user_split(data, test_size)
    cv = UserKFold(n_splits=k, shuffle=True, random_state=10701)
    results = cross_validate(algo, data, cv=cv, verbose=True)
    return trainset, testset, results



# Warm users
# Splits the data set into training and testing data, and run cross validationon training data
def split_and_cross_validate(algo, data=default_data, test_size=0.2, train_size=None, random_state=None, shuffle=True, measures=['rmse', 'mae'], cv=None, return_train_measures=False, n_jobs=1, pre_dispatch='2*n_jobs', verbose=False):
    train_set, test_set = train_test_split(data, test_size, train_size, random_state, shuffle)
    ret = cross_validate(algo, train_set, measures, cv, return_train_measures, n_jobs, pre_dispatch, verbose)
    return train_set, test_set, ret





#evaluation
def precision_at_k(predicted, relevant, k):
    predicted = predicted[:k]
    return len(set(predicted) & set(relevant)) / k if k else 0

def recall_at_k(predicted, relevant, k):
    predicted = predicted[:k]
    return len(set(predicted) & set(relevant)) / len(relevant) if relevant else 0

def ndcg_at_k(predicted, relevant, k):
    predicted = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(predicted):
        if p in relevant:
            dcg += 1 / np.log2(i + 2)  # rank starts at 1
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0

def test(trainset, testset, algo, relevance_threshold=4):
    test_users = set([uid for (uid, _, _) in testset])
    recs = {}

    # Iterate over unique users in the testset
    for uid in test_users:
        # Get the items that the user has already rated from the trainset
        known_items = set([iid for (iid, _) in trainset.ur[trainset.to_inner_uid(uid)]])
        all_items = set([iid for (uid, iid, _) in testset])
        candidates = [iid for iid in all_items if iid not in known_items]
        
        predictions = [(iid, algo.predict(uid, iid).est) for iid in candidates]
        
        top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
        recs[uid] = [iid for iid, _ in top_n]
    
    test_df = pd.DataFrame(testset, columns =['UserID', 'MovieID', 'Rating'])

    # Build a dictionary of relevant items per user
    ground_truth = (
        test_df[test_df['Rating'] >= relevance_threshold]
        .groupby('UserID')['MovieID']
        .apply(set)
        .to_dict()
    )
    
    results = []

    for uid in ground_truth:
        relevant_items = ground_truth.get(uid, set())
        
        pred = recs.get(uid, [])

        results.append({
            # 'UserID': uid,
            'Precision@10': precision_at_k(pred, relevant_items, 10),
            'Recall@10': recall_at_k(pred, relevant_items, 10),
            'NDCG@10': ndcg_at_k(pred, relevant_items, 10),
        })
        
    eval_df = pd.DataFrame(results)
    return eval_df
