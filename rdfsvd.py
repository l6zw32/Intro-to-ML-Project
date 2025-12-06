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


class RDFSVD(AlgoBase):
    """
    SVD with Regularization Differentiating Function
    """

    def __init__(
        self,
        n_users,
        n_items,
        n_factors: int = 15,
        n_epochs: int = 50,
        lr: float = 0.005,
        lr_bias: float | None = None,
        lr_latent: float | None = None,
        lmbda: float = 1.0,
        lmbda_p: float | None = None,
        lmbda_q: float | None = None,
        lmbda_u: float | None = None,
        lmbda_i: float | None = None,
        lr_shrink_rate: float = 0.9,
        method: str = "log",
        alpha: float = np.exp(1),
    ):
        super().__init__()

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

        # Parameters
        self.P = np.random.randn(self.n_users, self.n_factors)
        self.Q = np.random.randn(self.n_items, self.n_factors)
        self.bias_u = np.zeros(self.n_users)
        self.bias_i = np.zeros(self.n_items)

        self.method = method
        self.alpha = alpha

        # Counts
        self.n_user_rating: dict[int, int] | None = None
        self.n_item_rating: dict[int, int] | None = None

        # Precomputed differentiated regularization weights
        self.reg_bu: np.ndarray | None = None
        self.reg_bi: np.ndarray | None = None
        self.reg_pu: np.ndarray | None = None
        self.reg_qi: np.ndarray | None = None

    def _compute_n_user_item_rating(self, ratings):
        n_user_rating = collections.defaultdict(int)
        n_item_rating = collections.defaultdict(int)
        for (u, i, r) in ratings.all_ratings():
            n_user_rating[u] += 1
            n_item_rating[i] += 1
        self.n_user_rating = dict(n_user_rating)
        self.n_item_rating = dict(n_item_rating)

    def _precompute_reg_weights(self):
        """
        Precompute per-user and per-item differentiated regularization weights
        so we don't recalc them inside the innermost training loop.
        """
        assert self.n_user_rating is not None
        assert self.n_item_rating is not None

        self.reg_bu = np.zeros(self.n_users, dtype=float)
        self.reg_pu = np.zeros(self.n_users, dtype=float)
        self.reg_bi = np.zeros(self.n_items, dtype=float)
        self.reg_qi = np.zeros(self.n_items, dtype=float)

        # Helper: f(count) = 1/(count+alpha), or 1/sqrt, or 1/log
        def scale(count: int) -> float:
            denom = count + self.alpha
            if self.method == "linear":
                return 1.0 / denom
            elif self.method == "sqrt":
                return 1.0 / np.sqrt(denom)
            else:  # "log"
                return 1.0 / np.log(denom)

        # Users
        for u in range(self.n_users):
            c = self.n_user_rating.get(u, 0)
            s = scale(c)
            self.reg_bu[u] = self.lmbda_u * s
            self.reg_pu[u] = self.lmbda_p * s

        # Items
        for i in range(self.n_items):
            c = self.n_item_rating.get(i, 0)
            s = scale(c)
            self.reg_bi[i] = self.lmbda_i * s
            self.reg_qi[i] = self.lmbda_q * s

    def fit(self, ratings):
        super().fit(ratings)
        self.global_mean = ratings.global_mean

        # Count ratings and precompute regularization weights
        self._compute_n_user_item_rating(ratings)
        self._precompute_reg_weights()

        # Local aliases (small speedup)
        P = self.P
        Q = self.Q
        bu = self.bias_u
        bi = self.bias_i
        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        gmean = self.global_mean

        for epoch in range(self.n_epochs):
            epoch_shrink = self.lr_shrink_rate ** epoch
            lr_b = self.lr_bias * epoch_shrink
            lr_l = self.lr_latent * epoch_shrink

            for (u, i, r) in ratings.all_ratings():
                u = int(u)
                i = int(i)
                r = float(r)

                # Current params
                bu_u = bu[u]
                bi_i = bi[i]
                pu = P[u, :]
                qi = Q[i, :]

                # Fast inlined prediction
                pred = gmean + bu_u + bi_i + np.dot(pu, qi)
                err = r - pred

                # Regularization weights (precomputed)
                r_bu = reg_bu[u]
                r_bi = reg_bi[i]
                r_pu = reg_pu[u]
                r_qi = reg_qi[i]

                # Update biases
                bu[u] -= lr_b * (-err + r_bu * bu_u)
                bi[i] -= lr_b * (-err + r_bi * bi_i)

                # Update latent factors
                P[u, :] -= lr_l * (-err * qi + r_pu * pu)
                Q[i, :] -= lr_l * (-err * pu + r_qi * qi)

        # Write back (in case we used local aliases)
        self.P = P
        self.Q = Q
        self.bias_u = bu
        self.bias_i = bi

        return self

    def estimate(self, u: int, i: int):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        bu = self.bias_u[u] if known_user else 0.0
        bi = self.bias_i[i] if known_item else 0.0
        pu = self.P[u, :] if known_user else np.zeros(self.n_factors)
        qi = self.Q[i, :] if known_item else np.zeros(self.n_factors)

        return self.global_mean + bu + bi + float(np.dot(pu, qi))

    # These are unused but kept for completeness from the original code
    def _external_internal_id_mapping(self, ratings):
        self.eu2iu = {}
        self.iu2eu = {}
        self.ei2ii = {}
        self.ii2ei = {}
        for (eu, ei, r) in ratings:
            if eu not in self.eu2iu:
                iu = len(self.eu2iu)
                self.eu2iu[eu] = iu
                self.iu2eu[iu] = eu
            if ei not in self.ei2ii:
                ii = len(self.ei2ii)
                self.ei2ii[ei] = ii
                self.ii2ei[ii] = ei


if __name__ == "__main__":
    trainset, testset = cold_start_train()
    alg = RDFSVD(trainset.n_users, trainset.n_items)
    alg.fit(trainset)
    print("Training Complete")
    print(evaluate(testset, alg))

