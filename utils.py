# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 20:29:17 2025

Some helper functions for generating training and testing data
"""

import random
import math
from collections import defaultdict
from surprise import Dataset, Reader, Trainset, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd
import numpy as np

default_data = Dataset.load_builtin("ml-100k")

class UserKFold:
    """Splits the dataset into folds by users.
    Each fold contains a subset of users; validation set contains 1-frac of all ratings of those users.
    Train set contains all other ratings
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
def cold_start_train(data : Dataset =default_data, cold_start_user_portion=0.2, frac=None, n=5, random_seed=10701):
    """Splits data to training and testing sets based on users
    frac or n specifies how many ratings of each cold start user is revealed in training set. 
    They should not be specified at the same time. If this happens, frac takes priority. 

    Returns
    -------
    trainset : Trainset
        Training set. contains either frac of each cold start user's ratings or n ratings from each cold start user.
    testset : list[tuple[str, str, int]]
        Test set. Contains at most cold_start_user_portion of all users.

    """

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

        if frac is not None:
            revealed = user_ratings.sample(frac=frac, random_state=random_seed)
        elif len(user_ratings) > n:
            revealed = user_ratings.sample(n=n, random_state=random_seed)
        else: 
            continue
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
def user_split(data : Dataset =default_data, test_size=0.2):
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
def cold_start_cross_validate(algo, data : Dataset=default_data, trainset : Trainset | None =None, test_size=0.2 , k=5):
    """Cross validates by splitting data on users.
    if trainset is defined, trainset is used and returned, and testset is set to None. This renders data and test_size useless.
    Otherwise data is split into training and test set by on users. Test set will contain approximately test_size of all users.
    CURRENTLY DEPRECATED!!! DON'T USE!!! (Unless you define trainset)
    I will only fix it if necessary.

    Returns
    -------
    trainset : Trainset
        Train set.
    testset : list[tuple[str, str, int]]
        Test set. Contains test_size of all ratings.
    results : dict
        Results of running cross_validate on train_set.

    """
    testset = None
    if trainset == None:
        trainset, testset = user_split(data, test_size)
    cv = UserKFold(n_splits=k, shuffle=True, random_state=10701)
    results = cross_validate(algo, data, cv=cv, verbose=True)
    return trainset, testset, results



# Warm start
def split_and_cross_validate(algo, data : Dataset =default_data, test_size=0.2, train_size=None, random_state=None, shuffle=True, measures=['rmse', 'mae'], cv=None, return_train_measures=False, n_jobs=1, pre_dispatch='2*n_jobs', verbose=False):
    """Splits the data set into training and testing data, and run cross validation on training data
    
    Returns
    -------
    train_set : Trainset
        Train set.
    test_set : list[tuple[str, str, int]]
        Test set. Contains test_size of all ratings
    ret : dict
        Results of running cross_validate on train_set.

    """
    train_set, test_set = train_test_split(data, test_size, train_size, random_state, shuffle)
    ret = cross_validate(algo, train_set, measures, cv, return_train_measures, n_jobs, pre_dispatch, verbose)
    return train_set, test_set, ret





#evaluation
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user
    adopted from https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-compute-precision-k-and-recall-k
    """

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

# deprecated? I am not sure if it is bugged
def ndcg_at_k(predicted, relevant, k):
    predicted = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(predicted):
        if p in relevant:
            dcg += 1 / np.log2(i + 2)  # rank starts at 1
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0

def evaluate(testset : list[tuple[str, str, int]], algo, k=10, relevance_threshold=3.5):
    """Evaluates algo on the testset.
    Evaluation is done using the following metrics: RMSE, MAE, Precision at k, Recall at k
    Precision and recall are computed at the specified threshold for relevance.
    
    Returns
    -------
    rmse : float
        The Root Mean Squared Error of predictions.
    mae : float
        The Mean Absolute Error of predictions.
    precision : float
        Average precision at k across all users in the testset.
    recall : float
        Average recall at k across all users in the testset.

    """
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k, threshold=relevance_threshold)

    return accuracy.rmse(predictions), accuracy.mae(predictions), sum(prec for prec in precisions.values()) / len(precisions), sum(rec for rec in recalls.values()) / len(recalls)
            