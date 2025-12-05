# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 05:20:38 2025

"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
from utils import cold_start_train, evaluate
from ensemble import Ensemble
from cluster import Cluster

trainset, testset = cold_start_train()
alg = Ensemble()
#alg = Cluster()
alg.fit(trainset)
print(evaluate(testset, alg))