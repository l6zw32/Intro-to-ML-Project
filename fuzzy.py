import os
import sys
from typing import Dict, List, Tuple, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, AlgoBase
from sklearn.preprocessing import StandardScaler
from utils import cold_start_train, precision_recall_at_k, ndcg_at_k, evaluate

try:
    import skfuzzy as fuzz
except Exception as e:
    raise RuntimeError("scikit-fuzzy is required. Install with: pip install scikit-fuzzy") from e


# Reproducibility
# FUZZY_RANDOM_SEED = 10701
# np.random.seed(FUZZY_RANDOM_SEED)

# Data locations to probe
# DATA_DIR_CANDIDATES = [
#     "./Data"
# ]

# Core hyperparameters
FUZZY_RANDOM_SEED = 10701
FUZZY_NUM_CLUSTERS = 8
FUZZY_M = 2.0
FUZZY_MAXITER = 300
FUZZY_ERROR = 1e-5
FRIENDS_K = 20
COMBINATION_COEFF_C = 0.5
MIN_OVERLAP = 2
TOPN = 10
# DATA_DIR_CANDIDATES = ["./Data"]


class FuzzyAlgo(AlgoBase):
    # Hyperparameters (override in __init__ if desired)

    @dataclass
    class Model:
        user_id_order: List[int]
        sim_user: np.ndarray
        truth_df: pd.DataFrame
        scaler: StandardScaler
        cntr: np.ndarray
        u_train: np.ndarray
        sim_fuzzy: np.ndarray
        sim_combined: np.ndarray
        user_mean_rating: Dict[int, float]

        def __post_init__(self) -> None:
            self.user_index: Dict[int, int] = {uid: i for i, uid in enumerate(self.user_id_order)}

    def __init__(self,
                 n_clusters: int | None = None,
                 friends_k: int | None = None,
                 combo_c: float | None = None,
                 random_seed: int | None = None):
        super().__init__()
        # Allow simple overrides
        if n_clusters is not None:
            self.FUZZY_NUM_CLUSTERS = n_clusters
        else:
            self.FUZZY_NUM_CLUSTERS = FUZZY_NUM_CLUSTERS

        if friends_k is not None:
            self.FRIENDS_K = friends_k
        else:
            self.FRIENDS_K = FRIENDS_K

        if combo_c is not None:
            self.COMBINATION_COEFF_C = combo_c
        else:
            self.COMBINATION_COEFF_C = COMBINATION_COEFF_C

        if random_seed is not None:
            self.FUZZY_RANDOM_SEED = random_seed
        else:
            self.FUZZY_RANDOM_SEED = FUZZY_RANDOM_SEED

        np.random.seed(self.FUZZY_RANDOM_SEED)
        self.FUZZY_M = FUZZY_M
        self.FUZZY_ERROR = FUZZY_ERROR
        self.FUZZY_MAXITER = FUZZY_MAXITER
        self.MIN_OVERLAP = MIN_OVERLAP
        self.TOPN = TOPN

        self.model: FuzzyAlgo.Model | None = None
        self.train_df: pd.DataFrame | None = None
        self.train_by_user: Dict[int, Dict[int, float]] | None = None

        # Cache per user: uid -> (sim_combined, neighbor_idx_sorted, user_mean)
        self._user_cache: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}

    @staticmethod
    def _trainset_to_df(trainset) -> pd.DataFrame:
        rows = []
        for (inner_uid, inner_iid, rating) in trainset.all_ratings():
            uid = trainset.to_raw_uid(inner_uid)
            iid = trainset.to_raw_iid(inner_iid)
            rows.append((int(uid), int(iid), float(rating)))
        return pd.DataFrame(rows, columns=["UserID", "MovieID", "Rating"])

    # ---------- Training pipeline ----------
    def fit(self, trainset):
        super().fit(trainset)
        self.train_df = self._trainset_to_df(trainset)
        self.model = self._train_fuzzy_user_model(self.train_df)
        # Precompute ratings by user once
        self.train_by_user = self._ratings_by_user(self.train_df)
        # Clear cache in case fit is called again
        self._user_cache.clear()
        return self

    def _train_fuzzy_user_model(self, train_df: pd.DataFrame) -> Model:
        sim_user, user_id_order, user_mean_rating = self._build_user_based_similarity_surprise(train_df)
        truth_df, features = self._compute_truthfulness_features(train_df, user_id_order, sim_user)
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        cntr, u_train = self._run_fuzzy_cmeans(X)
        sim_fuzzy = self._cosine_sim_from_membership(u_train)
        sim_combined = self._combine_similarity(sim_user, sim_fuzzy)
        return FuzzyAlgo.Model(
            user_id_order=user_id_order,
            sim_user=sim_user,
            truth_df=truth_df,
            scaler=scaler,
            cntr=cntr,
            u_train=u_train,
            sim_fuzzy=sim_fuzzy,
            sim_combined=sim_combined,
            user_mean_rating=user_mean_rating,
        )

    def _build_user_based_similarity_surprise(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, List[int], Dict[int, float]]:
        """
        Build the user–movie rating matrix (users as rows, movies as columns, NaN for unrated),
        then compute the user–user Pearson similarity matrix with pairwise-complete observations.
        """
        pivot = train_df.pivot_table(index="UserID", columns="MovieID", values="Rating", aggfunc="mean")
        user_id_order = pivot.index.tolist()
        # Per-user mean rating over rated items
        user_mean_rating = pivot.mean(axis=1).to_dict()
        # User–user Pearson correlation (pairwise, ignoring NaNs)
        sim_df = pivot.T.corr(min_periods=2)
        sim = sim_df.fillna(0.0).to_numpy()
        return sim, user_id_order, user_mean_rating

    def _compute_truthfulness_features(self,
                                       train_df: pd.DataFrame,
                                       user_id_order: List[int],
                                       user_sim: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        activity = train_df.groupby("UserID").size().reindex(user_id_order).fillna(0).astype(float).values
        user_groups = train_df.groupby("UserID")
        probity_list: List[float] = []

        # pre-compute user mean ratings for friends score
        user_means_map: Dict[int, float] = user_groups["Rating"].mean().to_dict()

        for uid in user_id_order:
            grp = user_groups.get_group(uid) if uid in user_groups.groups else None
            if grp is None or grp.empty:
                probity_list.append(0.0)
                continue
            mu = grp["Rating"].mean()
            probity_list.append(float(np.sqrt(np.mean((grp["Rating"].values - mu) ** 2))))
        probity = np.array(probity_list, dtype=float)

        friends_scores: List[float] = []
        for i in range(len(user_id_order)):
            sims = user_sim[i].copy()
            sims[i] = -np.inf  # exclude self
            order = np.argsort(sims)[::-1]
            top_idx = [idx for idx in order if sims[idx] >= 0.0][: max(1, self.FRIENDS_K)]
            if not top_idx:
                friends_scores.append(0.0)
                continue
            neighbor_means = np.array(
                [user_means_map.get(user_id_order[idx], 0.0) for idx in top_idx],
                dtype=float
            )
            prefix_means = np.cumsum(neighbor_means) / (np.arange(len(neighbor_means)) + 1)
            friends_scores.append(float(np.max(prefix_means)))

        friends_scores = np.array(friends_scores, dtype=float)
        truth_df = pd.DataFrame(
            {"UserID": user_id_order, "activity": activity, "probity": probity, "friends": friends_scores}
        )
        features_matrix = truth_df[["activity", "probity", "friends"]].values
        return truth_df, features_matrix

    def _run_fuzzy_cmeans(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        data_T = features.T
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data_T,
            c=self.FUZZY_NUM_CLUSTERS,
            m=self.FUZZY_M,
            error=self.FUZZY_ERROR,
            maxiter=self.FUZZY_MAXITER,
            init=None,
            seed=self.FUZZY_RANDOM_SEED,
        )
        return cntr, u

    @staticmethod
    def _cosine_sim_from_membership(u: np.ndarray) -> np.ndarray:
        M = u.T
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        M_norm = M / norms
        return M_norm @ M_norm.T

    def _combine_similarity(self, sim_user: np.ndarray, sim_fuzzy: np.ndarray) -> np.ndarray:
        sim_user_pos = np.clip(sim_user, 0.0, 1.0)
        sim_fuzzy_pos = np.clip(sim_fuzzy, 0.0, 1.0)
        return self.COMBINATION_COEFF_C * sim_user_pos + (1.0 - self.COMBINATION_COEFF_C) * sim_fuzzy_pos

    # ---------- Estimation ----------
    def estimate(self, u, i):
        try:
            raw_uid = int(self.trainset.to_raw_uid(u))
        except Exception:
            raw_uid = int(str(u).replace("UKN__", ""))

        try:
            raw_iid = int(self.trainset.to_raw_iid(i))
        except Exception:
            raw_iid = int(str(i).replace("UKN__", ""))

        assert self.model is not None and self.train_df is not None and self.train_by_user is not None

        # Compute per-user similarities only once and cache them
        if raw_uid not in self._user_cache:
            revealed_df_user = self.train_df[self.train_df["UserID"] == raw_uid]
            sim_c, nbr_idx, user_mean, _ = self._compute_new_user_sims(
                raw_uid, revealed_df_user, self.train_df, self.model
            )
            self._user_cache[raw_uid] = (sim_c, nbr_idx, user_mean)
        else:
            sim_c, nbr_idx, user_mean = self._user_cache[raw_uid]

        _, est = self._estimate_single_item(
            raw_iid,
            sim_c,
            nbr_idx,
            user_mean,
            self.train_by_user,
            self.model.user_id_order,
            self.model.user_mean_rating,
            self.FRIENDS_K,
        )
        return float(est) if est else float(user_mean)

    @staticmethod
    def _ratings_by_user(train_df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
        by_user: Dict[int, Dict[int, float]] = {}
        for r in train_df.itertuples(index=False):
            by_user.setdefault(int(r.UserID), {})[int(r.MovieID)] = float(r.Rating)
        return by_user

    @staticmethod
    def _revealed_ratings_for_user(revealed_df: pd.DataFrame, uid: int) -> Dict[int, float]:
        return {
            int(r.MovieID): float(r.Rating)
            for r in revealed_df[revealed_df["UserID"] == uid].itertuples(index=False)
        }

    @staticmethod
    def _pearson_with_training(new_user_ratings: Dict[int, float],
                               train_by_user: Dict[int, Dict[int, float]],
                               min_overlap: int) -> Dict[int, float]:
        sims: Dict[int, float] = {}
        if not new_user_ratings:
            return sims
        for tuid, tr in train_by_user.items():
            common = set(new_user_ratings.keys()) & set(tr.keys())
            if len(common) < min_overlap:
                continue
            x_vals = [new_user_ratings[i] for i in common]
            y_vals = [tr[i] for i in common]
            x = np.array(x_vals, dtype=float) - np.mean(x_vals)
            y = np.array(y_vals, dtype=float) - np.mean(y_vals)
            denom = (np.linalg.norm(x) * np.linalg.norm(y))
            if denom <= 1e-12:
                continue
            sims[tuid] = float(np.dot(x, y) / denom)
        return sims

    def _compute_new_user_truthfulness(self,
                                       new_user_ratings: Dict[int, float],
                                       sims_to_training: Dict[int, float],
                                       user_mean_map: Dict[int, float],
                                       k: int) -> np.ndarray:
        activity = float(len(new_user_ratings))
        if activity == 0:
            probity = 0.0
        else:
            vals = np.array(list(new_user_ratings.values()), dtype=float)
            probity = float(np.sqrt(np.mean((vals - vals.mean()) ** 2)))

        if not sims_to_training:
            friends = 0.0
        else:
            pairs = sorted(
                [(u, s) for u, s in sims_to_training.items() if s is not None and s >= 0.0],
                key=lambda t: t[1],
                reverse=True,
            )
            top = pairs[: max(1, k)]
            if not top:
                friends = 0.0
            else:
                means = np.array([user_mean_map.get(u, 0.0) for (u, _) in top], dtype=float)
                prefix_means = np.cumsum(means) / (np.arange(len(means)) + 1)
                friends = float(np.max(prefix_means))

        return np.array([activity, probity, friends], dtype=float)

    def _cmeans_predict_membership(self, x_scaled: np.ndarray, cntr: np.ndarray) -> np.ndarray:
        data_T = x_scaled.T
        result = fuzz.cluster.cmeans_predict(
            data_T,
            cntr,
            m=self.FUZZY_M,
            error=self.FUZZY_ERROR,
            maxiter=self.FUZZY_MAXITER,
        )
        u = result[0] if isinstance(result, tuple) else result
        return u

    @staticmethod
    def _membership_cosine_similarity(u_train: np.ndarray, u_new: np.ndarray) -> np.ndarray:
        M_train = u_train.T
        M_new = u_new.T
        M_train = M_train / (np.linalg.norm(M_train, axis=1, keepdims=True) + 1e-12)
        M_new = M_new / (np.linalg.norm(M_new, axis=1, keepdims=True) + 1e-12)
        sims = (M_new @ M_train.T).ravel()
        return np.clip(sims, 0.0, 1.0)

    @staticmethod
    def _align_clip_user_sims(sims_user: Dict[int, float], user_id_order: List[int]) -> np.ndarray:
        """Align dict of user-based sims to training order and clamp negatives to zero."""
        return np.array(
            [max(0.0, sims_user.get(tuid, 0.0)) for tuid in user_id_order],
            dtype=float,
        )

    def _fuzzy_sims_with_new(self, model: Model, truth_new: np.ndarray) -> np.ndarray:
        """
        Compute fuzzy similarities by assigning the new user to existing clusters
        and comparing membership vectors to training users.
        """
        # Scale only the new user's features
        x_new_scaled = model.scaler.transform(truth_new.reshape(1, -1))
        # Predict membership for the new user using existing centers
        u_new = self._cmeans_predict_membership(x_new_scaled, model.cntr)
        # Similarity between new user and all training users in membership space
        return self._membership_cosine_similarity(model.u_train, u_new)

    def _compute_new_user_sims(self,
                               uid: int,
                               revealed_df: pd.DataFrame,
                               train_df: pd.DataFrame,
                               model: Model) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, Dict[int, float]]]:
        assert self.train_by_user is not None
        train_by_user = self.train_by_user

        # Build new user's ratings and initial user-based sims
        r_user = self._revealed_ratings_for_user(revealed_df, uid)
        sims_user = self._pearson_with_training(r_user, train_by_user, self.MIN_OVERLAP)

        # Truthfulness for the new user (activity, probity, friends)
        truth_new = self._compute_new_user_truthfulness(
            r_user, sims_user, model.user_mean_rating, self.FRIENDS_K
        )

        # Fuzzy sims using pre-trained cluster centers
        sim_fuzzy_new_vs_train = self._fuzzy_sims_with_new(model, truth_new)

        # Align and combine with user-based sims
        sim_user_new_vs_train = self._align_clip_user_sims(sims_user, model.user_id_order)
        sim_combined = np.clip(
            self.COMBINATION_COEFF_C * sim_user_new_vs_train
            + (1.0 - self.COMBINATION_COEFF_C) * sim_fuzzy_new_vs_train,
            0.0,
            1.0,
        )

        user_mean = (
            float(np.mean(list(r_user.values())))
            if r_user
            else float(np.mean(list(model.user_mean_rating.values())))
        )
        neighbor_idx_sorted = np.argsort(sim_combined)[::-1]
        return sim_combined, neighbor_idx_sorted, user_mean, train_by_user

    @staticmethod
    def _estimate_single_item(mid: int,
                              sim_combined: np.ndarray,
                              neighbor_idx_sorted: np.ndarray,
                              user_mean: float,
                              train_by_user: Dict[int, Dict[int, float]],
                              train_order: List[int],
                              user_mean_map: Dict[int, float],
                              top_k_neighbors: int) -> Tuple[int, float]:
        """Estimate a single item's rating for the new user using mean-centered neighbor aggregation."""
        weights: List[float] = []
        residuals: List[float] = []
        for idx in neighbor_idx_sorted[: max(1, top_k_neighbors)]:
            tuid = train_order[idx]
            tr = train_by_user.get(tuid)
            if not tr or mid not in tr:
                continue
            w = sim_combined[idx]
            if w <= 0:
                continue
            mu_u = user_mean_map.get(tuid)
            if mu_u is None:
                continue
            weights.append(w)
            residuals.append(tr[mid] - mu_u)

        if not weights:
            return int(mid), float(user_mean)

        w_arr = np.asarray(weights)
        r_arr = np.asarray(residuals)
        est = user_mean + float(
            np.dot(w_arr, r_arr) / (np.sum(np.abs(w_arr)) + 1e-12)
        )
        est = min(5.0, max(1.0, est))
        return int(mid), float(est)


def main() -> None:
    # Load Surprise built-in MovieLens 100K dataset
    data = Dataset.load_builtin("ml-100k")

    # Partial cold-start split using utils (Surprise)
    trainset, testset = cold_start_train(data, cold_start_user_portion=0.2, n=5, random_seed=FUZZY_RANDOM_SEED)

    # Train fuzzy user model via AlgoBase API
    fuzzy_algo = FuzzyAlgo(n_clusters = FUZZY_NUM_CLUSTERS, friends_k = FRIENDS_K, combo_c = COMBINATION_COEFF_C, random_seed = FUZZY_RANDOM_SEED).fit(trainset)
    print(f"Train ratings: {len(fuzzy_algo.train_df):,} | Test triples: {len(testset):,}")
    print("Fuzzy user model trained.")

    # Evaluate via Surprise-style API (AlgoBase)
    rmse, mae, precision, recall = evaluate(testset, fuzzy_algo, k=10, relevance_threshold=3.5)

    print("Evaluation on cold-start users (Surprise-style):")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  Precision@10: {precision:.4f}")
    print(f"  Recall@10:    {recall:.4f}")


if __name__ == "__main__":
    main()

