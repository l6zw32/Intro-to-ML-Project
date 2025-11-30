# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 17:53:52 2025

"""

from surprise import SVD, BaselineOnly, KNNBasic
from data_split import cold_start, test


# matrix factorisation baseline methods
baseline = BaselineOnly()
svd = SVD()

# collaborative filtering baseline method
knn = KNNBasic()

#partial cold start
trainset, testset = cold_start()
baseline.fit(trainset)
svd.fit(trainset)
knn.fit(trainset)

print(test(trainset, testset, baseline).mean())
print(test(trainset, testset, svd).mean())
print(test(trainset, testset, knn).mean())
