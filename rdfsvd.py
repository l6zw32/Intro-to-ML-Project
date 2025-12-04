# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 00:33:23 2025

adopted from https://github.com/ncu-dart/rdf . Modified the code to work with the scikit-surprise library
Chen, Hung-Hsuan, and Pu Chen. "Differentiating Regularization Weights--A Simple Mechanism to Alleviate Cold Start in Recommender Systems." ACM Transactions on Knowledge Discovery from Data (TKDD) 13.1 (2019).

"""

import collections
import numpy as np
from surprise import AlgoBase
from utils import cold_start_train, evaluate
from tqdm import tqdm

class RDFSVD(AlgoBase):
    """
    SVD with Regularization Differentiating Function
    """
    def __init__(
            self, n_users, n_items, n_factors=15, n_epochs=50,
            lr=.005, lr_bias=None, lr_latent=None,
            lmbda=1., lmbda_p=None, lmbda_q=None, lmbda_u=None, lmbda_i=None,
            lr_shrink_rate=.9, method="log", alpha=np.exp(1)):
        AlgoBase.__init__(self)

        self.n_users = n_users
        self.n_items = n_items
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.lr = lr
        self.lr_bias = lr if lr_bias is None else lr_bias
        self.lr_latent = lr if lr_latent is None else lr_latent
        self.lmbda = lmbda
        self.lmbda_p = lmbda if lmbda_p is None else lmbda_p
        self.lmbda_q = lmbda if lmbda_q is None else lmbda_q
        self.lmbda_u = lmbda if lmbda_u is None else lmbda_u
        self.lmbda_i = lmbda if lmbda_i is None else lmbda_i
        self.lr_shrink_rate = lr_shrink_rate
        self.P = np.random.randn(self.n_users, self.n_factors)
        self.Q = np.random.randn(self.n_items, self.n_factors)
        self.bias_u = np.zeros(self.n_users)
        self.bias_i = np.zeros(self.n_items)
        self.method = method
        self.alpha = alpha
        self.n_user_rating = None
        self.n_item_rating = None

    def fit(self, ratings):
        AlgoBase.fit(self, ratings)
        self.global_mean = ratings.global_mean
        self._compute_n_user_item_rating(ratings)

        for epoch in tqdm(range(self.n_epochs)):
            epoch_shrink = self.lr_shrink_rate ** epoch
            for (u, i, r) in ratings.all_ratings():
                #print(self.bu)
                err = r - self.estimate(u, i)
                r = float(r)
                bu = self.bias_u[u]
                bi = self.bias_i[i]
                pu = self.P[u, :]
                qi = self.Q[i, :]

                reg_bu = (
                    self.lmbda_u / (self.n_user_rating[u] + self.alpha)
                    ) if self.method == "linear" else (
                    self.lmbda_u / np.sqrt(self.n_user_rating[u] + self.alpha)
                    ) if self.method == "sqrt" else (
                    self.lmbda_u / np.log(self.n_user_rating[u] + self.alpha)
                    )
                reg_bi = (
                    self.lmbda_i / (self.n_item_rating[i] + self.alpha)
                    ) if self.method == "linear" else (
                    self.lmbda_i / np.sqrt(self.n_item_rating[i] + self.alpha)
                    ) if self.method == "sqrt" else (
                    self.lmbda_i / np.log(self.n_item_rating[i] + self.alpha)
                    )
                reg_pu = (
                    self.lmbda_p / (self.n_user_rating[u] + self.alpha)
                    ) if self.method == "linear" else (
                    self.lmbda_p / np.sqrt(self.n_user_rating[u] + self.alpha)
                    ) if self.method == "sqrt" else (
                    self.lmbda_p / np.log(self.n_user_rating[u] + self.alpha)
                    )
                reg_qi = (
                    self.lmbda_q / (self.n_item_rating[i] + self.alpha)
                    ) if self.method == "linear" else (
                    self.lmbda_q / np.sqrt(self.n_item_rating[i] + self.alpha)
                    ) if self.method == "sqrt" else (
                    self.lmbda_q / np.log(self.n_item_rating[i] + self.alpha)
                    )

                self.bias_u[u] -= (self.lr_bias * epoch_shrink) * (
                    -err + reg_bu * bu)
                self.bias_i[i] -= (self.lr_bias * epoch_shrink) * (
                    -err + reg_bi * bi)
                self.P[u, :] -= (self.lr_latent * epoch_shrink) * (
                    -err * qi + reg_pu * pu)
                self.Q[i, :] -= (self.lr_latent * epoch_shrink) * (
                    -err * pu + reg_qi * qi)

    def estimate(self, u : int, i : int):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        bu = self.bias_u[u] if known_user else 0
        bi = self.bias_i[i] if known_item else 0
        pu = self.P[u, :] if known_user else np.zeros(self.n_factors)
        qi = self.Q[i, :] if known_item else np.zeros(self.n_factors)
        return self.global_mean + bu + bi + np.dot(pu, qi)

    def _external_internal_id_mapping(self, ratings):
        for (eu, ei, r) in ratings:
            if eu not in self.eu2iu:
                iu = len(self.eu2iu)
                self.eu2iu[eu] = iu
                self.iu2eu[iu] = eu
            if ei not in self.ei2ii:
                ii = len(self.ei2ii)
                self.ei2ii[ei] = ii
                self.ii2ei[ii] = ei

    def _compute_n_user_item_rating(self, ratings):
        n_user_rating = collections.defaultdict(int)
        n_item_rating = collections.defaultdict(int)
        for (u, i, r) in ratings.all_ratings():
            n_user_rating[u] += 1
            n_item_rating[i] += 1
        self.n_user_rating = dict(n_user_rating)
        self.n_item_rating = dict(n_item_rating)


if __name__ == "__main__":
    trainset, testset = cold_start_train()
    print(trainset)
    alg = RDFSVD(trainset.n_users, trainset.n_items)
    #print(alg.bu)
    alg.fit(trainset)
    print("Training Complete")
    print(alg.test(testset))
