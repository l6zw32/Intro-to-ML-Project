# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 06:46:27 2025

"""


from surprise import AlgoBase, Trainset, SVD
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


from surprise import AlgoBase, Trainset, SVD
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


class Cluster(AlgoBase):
    def __init__(self, n: int = 10, random_seed: int = 10701):
        super().__init__()
        self.n = n
        self.random_seed = random_seed

        self.train_df: pd.DataFrame | None = None
        self.svd: SVD | None = None
        self.kmeans: KMeans | None = None

        # cluster_id -> {MovieID: mean_rating}
        self.cluster_top: dict[int, dict[int, float]] = {}

        # inner_uid -> cluster_id
        self.user_cluster: np.ndarray | None = None

        # raw_uid -> {raw_iid: rating}
        self.train_by_user: dict[int, dict[int, float]] = {}

    def fit(self, trainset: Trainset):
        # Standard Surprise init
        super().fit(trainset)

        # Build DataFrame of (raw_uid, raw_iid, rating)
        train = [
            (trainset.to_raw_uid(uid),
             trainset.to_raw_iid(iid),
             rating)
            for (uid, iid, rating) in trainset.all_ratings()
        ]
        self.train_df = pd.DataFrame(train, columns=["UserID", "MovieID", "Rating"])

        # Build a fast user->items dict once
        self.train_by_user = {}
        for r in self.train_df.itertuples(index=False):
            uid = int(r.UserID)
            iid = int(r.MovieID)
            rating = float(r.Rating)
            self.train_by_user.setdefault(uid, {})[iid] = rating

        # Fit SVD once
        self.svd = SVD()
        self.svd.fit(trainset)

        # SVD user factors: row index == inner_uid
        user_factors = self.svd.pu  # shape: (n_users, n_factors)

        # Cluster in latent space
        self.kmeans = KMeans(n_clusters=self.n, random_state=self.random_seed)
        self.kmeans.fit(user_factors)

        # For fast lookup in estimate: cluster per inner user id
        self.user_cluster = self.kmeans.labels_.copy()

        # For each cluster, compute mean rating per item over users in that cluster
        self.cluster_top = {}
        n_users = user_factors.shape[0]

        # Build mapping: cluster_id -> set of raw_uids
        cluster_to_uids: dict[int, list[int]] = {}
        for inner_uid in range(n_users):
            raw_uid = int(trainset.to_raw_uid(inner_uid))
            c_id = int(self.user_cluster[inner_uid])
            cluster_to_uids.setdefault(c_id, []).append(raw_uid)

        for cluster_id, cluster_users in cluster_to_uids.items():
            cluster_ratings = self.train_df[self.train_df["UserID"].isin(cluster_users)]
            top_items = (
                cluster_ratings
                .groupby("MovieID")["Rating"]
                .mean()
                .to_dict()
            )
            # store as dict[int, float]
            self.cluster_top[cluster_id] = {int(k): float(v) for k, v in top_items.items()}

        return self

    def estimate(self, u, i):
        # If item/user unseen in training, back off to global mean
        if not self.trainset.knows_user(u) or not self.trainset.knows_item(i):
            return self.trainset.global_mean

        assert self.user_cluster is not None
        assert self.cluster_top is not None
        assert self.train_df is not None
        assert self.train_by_user is not None

        # Map to raw ids
        raw_uid = int(self.trainset.to_raw_uid(u))
        raw_iid = int(self.trainset.to_raw_iid(i))

        # Cluster is precomputed by inner user id
        cluster_id = int(self.user_cluster[u])

        # Cluster-level mean for this item if available
        cluster_items = self.cluster_top.get(cluster_id, {})
        if raw_iid in cluster_items:
            return cluster_items[raw_iid]

        # Otherwise, if the user has rated this item in train, use that
        user_ratings = self.train_by_user.get(raw_uid, {})
        if raw_iid in user_ratings:
            return user_ratings[raw_iid]

        # Fallback: global mean
        return self.trainset.global_mean
