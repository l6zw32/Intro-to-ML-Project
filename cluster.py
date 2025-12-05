# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 06:46:27 2025

"""


from surprise import AlgoBase, Trainset, SVD
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


class Cluster(AlgoBase):
    def __init__(self, n=10,random_seed=10701):
        AlgoBase.__init__(self)
        self.n=n
        self.random_seed=random_seed
        
    def fit(self, trainset : Trainset):
        # Extract user factors from SVD
        AlgoBase.fit(self, trainset)
        train = [(trainset.to_raw_uid(uid), trainset.to_raw_iid(iid), rating) for (uid, iid, rating) in trainset.all_ratings()]
        self.train_df = pd.DataFrame(train, columns=['UserID', 'MovieID', 'Rating'])
        self.svd = SVD()
        self.svd.fit(trainset)
        user_factors = self.svd.pu  # shape: (n_users, n_factors)
        user_ids = [trainset.to_raw_uid(i) for i in range(len(user_factors))]

        user_df = pd.DataFrame(user_factors, index=user_ids)
    
        # Fit k-means
        self.kmeans = KMeans(n_clusters=self.n, random_state=self.random_seed)
        user_df['Cluster'] = self.kmeans.fit_predict(user_df)

        # For each cluster, compute Top-N
        self.cluster_top = {}
        for cluster_id in user_df['Cluster'].unique():
            cluster_users = user_df[user_df['Cluster'] == cluster_id].index
            cluster_ratings = self.train_df[self.train_df['UserID'].isin(cluster_users)]

            top_items = (cluster_ratings
                 .groupby('MovieID')['Rating']
                 .mean()
                 .to_dict())
    
            self.cluster_top[cluster_id] = top_items
        


        
    def compute_user_vector(self, uid):
        user_data = self.train_df[self.train_df['UserID'] == uid]
    
        q_list, r_list = [], []
        for _, row in user_data.iterrows():
            try:
                inner_iid = self.svd.trainset.to_inner_iid(row['MovieID'])
                q_list.append(self.svd.qi[inner_iid])
                r_list.append(row['Rating'])
            except:
                continue  # skip items not in training

        if not q_list:
            return None

        Q = np.stack(q_list)
        r = np.array(r_list).reshape(-1, 1)

        # Solve for user vector p in least-squares sense
        p = np.linalg.pinv(Q.T @ Q) @ Q.T @ r
        return p.ravel()
    
    def estimate(self, u, i):
        if not self.trainset.knows_item(i) or not self.trainset.knows_user(u):
            return self.trainset.global_mean
        uid = self.trainset.to_raw_uid(u)
        iid = self.trainset.to_raw_iid(i)
        vec = self.compute_user_vector(uid)
        if vec is not None:
            cluster = self.kmeans.predict([vec])[0]
            if cluster is None:
                return self.trainset.global_mean
            ret = self.cluster_top.get(cluster, [])
            if iid in ret:
                return self.cluster_top[cluster][iid]
            elif iid in set(self.train_df[self.train_df['UserID'] == uid]['MovieID']):
                return self.train_df[(self.train_df['UserID'] == uid) & (self.train_df['MovieID'] == iid)]['Rating']
        return self.trainset.global_mean