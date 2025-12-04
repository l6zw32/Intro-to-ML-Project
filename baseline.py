# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 17:53:52 2025

"""

from surprise import SVD, BaselineOnly, KNNBasic
from utils import cold_start_train, evaluate 


# matrix factorisation baseline methods
baseline = BaselineOnly()
svd = SVD()

# collaborative filtering baseline method
knn = KNNBasic()

#partial cold start
trainset, testset = cold_start_train()
# cold_start_cross_validate(baseline)
# cold_start_cross_validate(svd)
# cold_start_cross_validate(knn)


baseline.fit(trainset)
svd.fit(trainset)
knn.fit(trainset)


print(evaluate(testset, baseline))
print(evaluate(testset, svd))
print(evaluate(testset, knn))
