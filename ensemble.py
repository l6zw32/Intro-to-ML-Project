# -*- coding: utf-8 -*-
"""
Ensemble of RDFSVD, FuzzyAlgo, and Cluster with bagging.
"""

import numpy as np
import pandas as pd

from rdfsvd import RDFSVD
from fuzzy import FuzzyAlgo
from cluster import Cluster

from surprise import Dataset, Reader, AlgoBase, Trainset

FUZZY_RANDOM_SEED = 10701
FUZZY_NUM_CLUSTERS = 8
FRIENDS_K = 20
COMBINATION_COEFF_C = 0.5


class Ensemble(AlgoBase):
    def __init__(self, random_state: int = 10701, k: int = 3):
        super().__init__()
        self.random_state = random_state
        self.k = k
        self.models: list[AlgoBase] = []

    def fit(self, trainset: Trainset):
        super().fit(trainset)

        # Cache ensemble global mean
        self.global_mean = trainset.global_mean

        # Convert trainset to DataFrame once
        rating_rows = [
            (trainset.to_raw_uid(uid),
             trainset.to_raw_iid(iid),
             float(r))
            for (uid, iid, r) in trainset.all_ratings()
        ]
        ratings = pd.DataFrame(rating_rows,
                               columns=["UserID", "ItemID", "Rating"])

        # Use proper rating scale from Surprise trainset
        min_r, max_r = trainset.rating_scale
        reader = Reader(rating_scale=(min_r, max_r))

        n = len(ratings)
        rng_seed = self.random_state
        self.models = []

        # Helper to build a bootstrapped Surprise trainset
        def bootstrap_trainset(seed: int) -> Trainset:
            sample_df = ratings.sample(
                n, replace=True, random_state=seed
            )
            data = Dataset.load_from_df(
                sample_df[["UserID", "ItemID", "Rating"]],
                reader
            )
            return data.build_full_trainset()

        # k RDFSVD models
        for _ in range(self.k):
            sample_trainset = bootstrap_trainset(rng_seed)
            rng_seed += 1

            rdf = RDFSVD(sample_trainset.n_users, sample_trainset.n_items)
            rdf.fit(sample_trainset)
            self.models.append(rdf)

        # k FuzzyAlgo models
        for _ in range(self.k):
            sample_trainset = bootstrap_trainset(rng_seed)
            rng_seed += 1

            fuzzy = FuzzyAlgo(
                n_clusters=FUZZY_NUM_CLUSTERS,
                friends_k=FRIENDS_K,
                combo_c=COMBINATION_COEFF_C,
                random_seed=FUZZY_RANDOM_SEED,
            )
            fuzzy.fit(sample_trainset)
            self.models.append(fuzzy)

        # k Cluster models
        for _ in range(self.k):
            sample_trainset = bootstrap_trainset(rng_seed)
            rng_seed += 1

            cl = Cluster()
            cl.fit(sample_trainset)
            self.models.append(cl)

        # Update internal RNG state
        self.random_state = rng_seed

        return self

    def estimate(self, u, i):
        # If ensemble trainset doesn't know item/user, just back off
        if not self.trainset.knows_item(i) or not self.trainset.knows_user(u):
            return self.global_mean

        raw_uid = self.trainset.to_raw_uid(u)
        raw_iid = self.trainset.to_raw_iid(i)

        preds = []

        for m in self.models:
            ts = m.trainset

            # Map raw IDs into the model's own trainset space
            try:
                inner_u = ts.to_inner_uid(raw_uid)
                inner_i = ts.to_inner_iid(raw_iid)
            except ValueError:
                # Model never saw this user or item
                preds.append(ts.global_mean)
                continue

            # Call model's estimate directly (cheaper than predict())
            try:
                est = m.estimate(inner_u, inner_i)
            except Exception:
                # If something goes wrong, be conservative
                est = ts.global_mean
            preds.append(float(est))

        if not preds:
            return self.global_mean

        return float(np.mean(preds))

if __name__ == "__main__":
    from utils import cold_start_train, evaluate
    import time
    start = time.time()
    trainset, testset = cold_start_train(n=5)
    alg = Ensemble()
    alg.fit(trainset)
    print(evaluate(testset, alg))
    end = time.time()
    print("Time taken:", end - start, "seconds")