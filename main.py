# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 05:20:38 2025

"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
from utils import cold_start_train, evaluate
from ensemble import Ensemble
from cluster import Cluster
from rdfsvd import RDFSVD
from fuzzy import FuzzyAlgo
from surprise import SVD, BaselineOnly, KNNBasic


# matrix factorisation baseline methods
baseline = BaselineOnly()
svd = SVD()

# collaborative filtering baseline method
knn = KNNBasic()

#partial cold start
trainset, testset = cold_start_train()

print("Baseline Models")
baseline.fit(trainset)
svd.fit(trainset)
knn.fit(trainset)
print("Baseline Models training complete")


print("RMSE \t MAE \t Precision@10\t Recall@10")
print('BaselineOnly', evaluate(testset, baseline))
print('SVD', evaluate(testset, svd))
print('kNN, k=40',evaluate(testset, knn))

print("----------------------------")
print("Basic Models")
print()
print("Training RDFSVD")
rdfsvd = RDFSVD(trainset.n_users, trainset.n_items)
rdfsvd.fit(trainset)
print("Training complete")
print("RMSE \t MAE \t Precision@10\t Recall@10")
print('RDFSVD', evaluate(testset, rdfsvd))
print()
print("Training Cluster")
c = Cluster(trainset.n_users, trainset.n_items)
c.fit(trainset)
print("Training complete")
print("RMSE \t MAE \t Precision@10\t Recall@10")
print('Cluster', evaluate(testset, c))
print()
print("Training Fuzzy")
fuzzy = FuzzyAlgo(trainset.n_users, trainset.n_items)
fuzzy.fit(trainset)
print("Training complete")
print("RMSE \t MAE \t Precision@10\t Recall@10")
print('RDFSVD', evaluate(testset, fuzzy))

print("----------------------------")
print('Ensemble: each basic model applied 3 times, with bagging')
alg = Ensemble()
alg.fit(trainset)
print("Training complete")
print("RMSE \t MAE \t Precision@10\t Recall@10")
print('Ensemble', evaluate(testset, alg))
