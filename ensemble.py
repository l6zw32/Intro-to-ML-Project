# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 05:12:35 2025

"""

import numpy as np
import pandas as pd
from rdfsvd import RDFSVD

#from fuzzy import *

from cluster import Cluster

from surprise import Dataset, Reader, AlgoBase, Trainset


class Ensemble(AlgoBase):
    def __init__(self, random_state=10701, k=3):
        AlgoBase.__init__(self)
        self.random_state=random_state
        self.k=k
        
    def fit(self, trainset : Trainset):
        AlgoBase.fit(self, trainset)
        n = trainset.n_ratings
        ratings = pd.DataFrame([(trainset.to_raw_uid(uid), trainset.to_raw_iid(iid), rating) for (uid, iid, rating) in trainset.all_ratings()], columns=["UserID", "ItemID", "Rating"])
        reader = Reader()
        self.models = []
        
        for i in range(self.k):
            sample_df=ratings.sample(n, replace=True, random_state=self.random_state)
            sample_data = Dataset.load_from_df(sample_df[["UserID", "ItemID", "Rating"]], reader).build_full_trainset()
            model = RDFSVD(sample_data.n_users, sample_data.n_items)
            self.models.append(model)
            model.fit(sample_data)
            self.random_state+=1
        # self.fuzzy = ???
        # self.fuzzy.fit(trainset)
        
        for i in range(self.k):
            sample_df=ratings.sample(n, replace=True, random_state=self.random_state)
            sample_data = Dataset.load_from_df(sample_df[["UserID", "ItemID", "Rating"]], reader).build_full_trainset()
            model = Cluster()
            self.models.append(model)
            model.fit(sample_data)
            self.random_state+=1
            
        
    def estimate(self, u, i):
        if not self.trainset.knows_item(i):
            return self.trainset.global_mean
        preds = []
        for m in self.models:
            preds.append(m.predict(self.trainset.to_raw_uid(u), self.trainset.to_raw_iid(i)).est)
        return float(np.mean(preds))