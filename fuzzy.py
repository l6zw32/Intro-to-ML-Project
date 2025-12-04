import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from sklearn.preprocessing import StandardScaler
from data_split import precision_at_k, recall_at_k, ndcg_at_k

try:
    import skfuzzy as fuzz
except Exception as e:
    raise RuntimeError("scikit-fuzzy is required. Install with: pip install scikit-fuzzy") from e


# Reproducibility
FUZZY_RANDOM_SEED = 10701
np.random.seed(FUZZY_RANDOM_SEED)

# Data locations to probe
DATA_DIR_CANDIDATES = [
    "./Data"
]

# Core hyperparameters
FUZZY_NUM_CLUSTERS = 8
FUZZY_M = 2.0  # fuzzifier
FUZZY_MAXITER = 300
FUZZY_ERROR = 1e-5
FRIENDS_K = 20
COMBINATION_COEFF_C = 0.5  # weight on Pearson user-based sim (in [0,1])
MIN_OVERLAP = 2  # min common movies to compute Pearson with new users
TOPN = 10


def _first_existing_path(candidates: List[str]) -> str:
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("No valid data directory found among: " + ", ".join(candidates))


def load_movielens_dfs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = _first_existing_path(DATA_DIR_CANDIDATES)

    # Use MovieLens 1M format
    sep = "::"
    rating_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    user_cols = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    movie_cols = ["MovieID", "Title", "Genres"]

    ratings_path = os.path.join(data_dir, "ratings.dat")
    users_path = os.path.join(data_dir, "users.dat")
    movies_path = os.path.join(data_dir, "movies.dat")

    ratings = pd.read_csv(ratings_path, sep=sep, engine="python", names=rating_cols)
    users = pd.read_csv(users_path, sep=sep, engine="python", names=user_cols)
    movies = pd.read_csv(movies_path, sep=sep, engine="python", names=movie_cols, encoding="latin-1")

    ratings["UserID"] = ratings["UserID"].astype(int)
    ratings["MovieID"] = ratings["MovieID"].astype(int)
    ratings["Rating"] = ratings["Rating"].astype(float)

    return ratings, users, movies


def split_cold_start(ratings: pd.DataFrame, seed: int = FUZZY_RANDOM_SEED,
                     cold_frac: float = 0.2, revealed_per_user: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate_users = (
        ratings.groupby("UserID").size().rename("count").reset_index()
    )
    candidate_users = candidate_users[candidate_users["count"] >= 2 * revealed_per_user]

    sampled = candidate_users["UserID"].sample(frac=cold_frac, random_state=seed)

    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for uid in sampled:
        user_r = ratings[ratings["UserID"] == uid]
        revealed = user_r.sample(n=revealed_per_user, random_state=seed)
        held_out = user_r.drop(revealed.index)
        train_parts.append(revealed)
        test_parts.append(held_out)

    remaining = ratings[~ratings["UserID"].isin(sampled)]
    train_parts.append(remaining)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def build_user_based_similarity_surprise(train_df: pd.DataFrame) -> Tuple[np.ndarray, List[int], Dict[int, float]]:
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["UserID", "MovieID", "Rating"]], reader)
    trainset = data.build_full_trainset()

    algo = KNNBasic(sim_options={"name": "pearson", "user_based": True})
    algo.fit(trainset)

    sim = algo.sim  # (n_users x n_users)
    inner_to_raw = {i: trainset.to_raw_uid(i) for i in range(trainset.n_users)}
    user_id_order = [inner_to_raw[i] for i in range(trainset.n_users)]

    user_mean_rating: Dict[int, float] = train_df.groupby("UserID")["Rating"].mean().to_dict()
    return sim, user_id_order, user_mean_rating


def compute_truthfulness_features(train_df: pd.DataFrame,
                                  user_id_order: List[int],
                                  user_sim: np.ndarray,
                                  knn_k: int = FRIENDS_K,
                                  friends_score_method: str = "max_prefix_mean") -> Tuple[pd.DataFrame, np.ndarray]:
    activity = train_df.groupby("UserID").size().reindex(user_id_order).fillna(0).astype(float).values

    user_groups = train_df.groupby("UserID")
    probity_list: List[float] = []
    for uid in user_id_order:
        grp = user_groups.get_group(uid) if uid in user_groups.groups else None
        if grp is None or grp.empty:
            probity_list.append(0.0)
            continue
        mu = grp["Rating"].mean()
        probity = float(np.sqrt(np.mean((grp["Rating"].values - mu) ** 2)))
        probity_list.append(probity)
    probity = np.array(probity_list, dtype=float)

    friends_scores: List[float] = []
    for i, _ in enumerate(user_id_order):
        sims = user_sim[i].copy()
        sims[i] = -np.inf  # drop self
        sims = np.where(sims < 0, 0.0, sims)  # keep non-negative
        nn_idx = np.argsort(sims)[::-1][:max(1, knn_k)]
        nn_vals = sims[nn_idx]
        if friends_score_method == "max_prefix_mean":
            prefix_means = np.cumsum(nn_vals) / (np.arange(len(nn_vals)) + 1)
            friends_scores.append(float(np.max(prefix_means)))
        else:
            friends_scores.append(float(np.mean(nn_vals)))
    friends_scores = np.array(friends_scores, dtype=float)

    truth_df = pd.DataFrame({
        "UserID": user_id_order,
        "activity": activity,
        "probity": probity,
        "friends": friends_scores,
    })
    features_matrix = truth_df[["activity", "probity", "friends"]].values
    return truth_df, features_matrix


def run_fuzzy_cmeans(features: np.ndarray,
                     n_clusters: int = FUZZY_NUM_CLUSTERS,
                     m: float = FUZZY_M,
                     maxiter: int = FUZZY_MAXITER,
                     error: float = FUZZY_ERROR,
                     seed: int = FUZZY_RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray]:
    data_T = features.T  # (n_features, n_samples)
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_T, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None, seed=seed
    )
    return cntr, u


def cosine_sim_from_membership(u: np.ndarray) -> np.ndarray:
    M = u.T  # (n_users, n_clusters)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    M_norm = M / norms
    return M_norm @ M_norm.T


def combine_similarity(sim_user: np.ndarray, sim_fuzzy: np.ndarray, c: float = COMBINATION_COEFF_C) -> np.ndarray:
    sim_user_pos = np.clip(sim_user, 0.0, 1.0)
    sim_fuzzy_pos = np.clip(sim_fuzzy, 0.0, 1.0)
    return c * sim_user_pos + (1.0 - c) * sim_fuzzy_pos


class FuzzyUserModel:
    def __init__(self, user_id_order: List[int], sim_user: np.ndarray,
                 truth_df: pd.DataFrame, scaler: StandardScaler,
                 cntr: np.ndarray, u_train: np.ndarray,
                 sim_fuzzy: np.ndarray, sim_combined: np.ndarray,
                 user_mean_rating: Dict[int, float]):
        self.user_id_order = user_id_order
        self.user_index = {uid: i for i, uid in enumerate(user_id_order)}
        self.sim_user = sim_user
        self.truth_df = truth_df
        self.scaler = scaler
        self.cntr = cntr
        self.u_train = u_train
        self.sim_fuzzy = sim_fuzzy
        self.sim_combined = sim_combined
        self.user_mean_rating = user_mean_rating


def train_fuzzy_user_model(train_df: pd.DataFrame) -> FuzzyUserModel:
    sim_user, user_id_order, user_mean_rating = build_user_based_similarity_surprise(train_df)

    truth_df, features = compute_truthfulness_features(
        train_df, user_id_order, sim_user, knn_k=FRIENDS_K, friends_score_method="max_prefix_mean"
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    cntr, u_train = run_fuzzy_cmeans(X, n_clusters=FUZZY_NUM_CLUSTERS,
                                     m=FUZZY_M, maxiter=FUZZY_MAXITER, error=FUZZY_ERROR,
                                     seed=FUZZY_RANDOM_SEED)

    sim_fuzzy = cosine_sim_from_membership(u_train)
    sim_combined = combine_similarity(sim_user, sim_fuzzy, c=COMBINATION_COEFF_C)

    return FuzzyUserModel(
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


def _ratings_by_user(train_df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    by_user: Dict[int, Dict[int, float]] = {}
    for r in train_df.itertuples(index=False):
        by_user.setdefault(int(r.UserID), {})[int(r.MovieID)] = float(r.Rating)
    return by_user


def pearson_with_training(new_user_ratings: Dict[int, float],
                          train_by_user: Dict[int, Dict[int, float]],
                          min_overlap: int = MIN_OVERLAP) -> Dict[int, float]:
    sims: Dict[int, float] = {}
    if not new_user_ratings:
        return sims

    for tuid, tr in train_by_user.items():
        common = set(new_user_ratings.keys()) & set(tr.keys())
        if len(common) < min_overlap:
            continue
        x = np.array([new_user_ratings[i] for i in common], dtype=float)
        y = np.array([tr[i] for i in common], dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y))
        if denom <= 1e-12:
            continue
        sims[tuid] = float(np.dot(x, y) / denom)
    return sims


def compute_new_user_truthfulness(new_user_ratings: Dict[int, float],
                                  sims_to_training: Dict[int, float],
                                  k: int = FRIENDS_K,
                                  method: str = "max_prefix_mean") -> np.ndarray:
    activity = float(len(new_user_ratings))
    if activity == 0:
        probity = 0.0
    else:
        vals = np.array(list(new_user_ratings.values()), dtype=float)
        probity = float(np.sqrt(np.mean((vals - vals.mean()) ** 2)))

    if not sims_to_training:
        friends = 0.0
    else:
        sims = np.array([max(0.0, s) for s in sims_to_training.values()], dtype=float)
        sims.sort()
        sims = sims[::-1][:max(1, k)]
        if method == "max_prefix_mean":
            prefix_means = np.cumsum(sims) / (np.arange(len(sims)) + 1)
            friends = float(np.max(prefix_means))
        else:
            friends = float(np.mean(sims))

    return np.array([activity, probity, friends], dtype=float)


def cmeans_predict_membership(x_scaled: np.ndarray, cntr: np.ndarray, m: float = FUZZY_M,
                              error: float = FUZZY_ERROR, maxiter: int = FUZZY_MAXITER) -> np.ndarray:
    data_T = x_scaled.T
    result = fuzz.cluster.cmeans_predict(data_T, cntr, m=m, error=error, maxiter=maxiter)
    if isinstance(result, tuple):
        u = result[0]
    else:
        u = result
    return u  # (n_clusters, n_samples)


def membership_cosine_similarity(u_train: np.ndarray, u_new: np.ndarray) -> np.ndarray:
    M_train = u_train.T  # (n_train, n_clusters)
    M_new = u_new.T      # (1, n_clusters)
    M_train = M_train / (np.linalg.norm(M_train, axis=1, keepdims=True) + 1e-12)
    M_new = M_new / (np.linalg.norm(M_new, axis=1, keepdims=True) + 1e-12)
    sims = (M_new @ M_train.T).ravel()
    return np.clip(sims, 0.0, 1.0)


def predict_ratings_for_user(uid: int,
                             revealed_df: pd.DataFrame,
                             train_df: pd.DataFrame,
                             model: FuzzyUserModel,
                             movies_all: np.ndarray,
                             top_k_neighbors: int = FRIENDS_K) -> List[Tuple[int, float]]:
    train_by_user = _ratings_by_user(train_df)
    r_user = {int(r.MovieID): float(r.Rating) for r in revealed_df[revealed_df["UserID"] == uid].itertuples(index=False)}

    sims_user = pearson_with_training(r_user, train_by_user, min_overlap=MIN_OVERLAP)
    truth_new = compute_new_user_truthfulness(r_user, sims_user, k=FRIENDS_K, method="max_prefix_mean")

    x_scaled = model.scaler.transform(truth_new.reshape(1, -1))
    u_new = cmeans_predict_membership(x_scaled, model.cntr, m=FUZZY_M, error=FUZZY_ERROR, maxiter=FUZZY_MAXITER)

    sim_fuzzy_new_vs_train = membership_cosine_similarity(model.u_train, u_new)

    train_order = model.user_id_order
    sim_user_new_vs_train = np.array([max(0.0, sims_user.get(tuid, 0.0)) for tuid in train_order], dtype=float)

    sim_combined = COMBINATION_COEFF_C * sim_user_new_vs_train + (1.0 - COMBINATION_COEFF_C) * sim_fuzzy_new_vs_train
    sim_combined = np.clip(sim_combined, 0.0, 1.0)

    user_mean = np.mean(list(r_user.values())) if r_user else np.mean(list(model.user_mean_rating.values()))
    neighbor_idx_sorted = np.argsort(sim_combined)[::-1]

    predictions: List[Tuple[int, float]] = []
    revealed_items = set(r_user.keys())
    candidates = [mid for mid in movies_all if mid not in revealed_items]
    idx_to_uid = train_order

    for mid in candidates:
        weights: List[float] = []
        residuals: List[float] = []
        for idx in neighbor_idx_sorted[:max(1, top_k_neighbors)]:
            tuid = idx_to_uid[idx]
            tr = train_by_user.get(tuid)
            if tr is None or mid not in tr:
                continue
            w = sim_combined[idx]
            if w <= 0:
                continue
            mv = model.user_mean_rating.get(tuid, None)
            if mv is None:
                continue
            weights.append(w)
            residuals.append(tr[mid] - mv)
        if not weights:
            est = float(user_mean)
        else:
            w = np.array(weights, dtype=float)
            r = np.array(residuals, dtype=float)
            denom = np.sum(np.abs(w)) + 1e-12
            est = float(user_mean + np.dot(w, r) / denom)
            est = min(5.0, max(1.0, est))
        predictions.append((mid, est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:TOPN]

def main() -> None:
    ratings, users, movies = load_movielens_dfs()
    train_f, test_f = split_cold_start(ratings, seed=FUZZY_RANDOM_SEED, cold_frac=0.2, revealed_per_user=5)

    print(f"Train ratings: {len(train_f):,} | Test ratings (held-out): {len(test_f):,}")
    model = train_fuzzy_user_model(train_f)
    print("Fuzzy user model trained.")

    all_movies = ratings["MovieID"].unique()
    all_movies.sort()

    revealed_f = train_f[train_f["UserID"].isin(test_f["UserID"].unique())]
    ground_truth_map = test_f.groupby("UserID")["MovieID"].apply(list).to_dict()

    results = []
    user_recs_fuzzy: Dict[int, List[int]] = {}
    for uid in test_f["UserID"].unique():
        recs = predict_ratings_for_user(uid, revealed_f, train_f, model, all_movies, top_k_neighbors=FRIENDS_K)
        top_movie_ids = [m for m, _ in recs]
        user_recs_fuzzy[uid] = top_movie_ids
        relevant = ground_truth_map.get(uid, [])
        results.append({
            "UserID": uid,
            "Precision@10_Fuzzy": precision_at_k(top_movie_ids, relevant, 10),
            "Recall@10_Fuzzy": recall_at_k(top_movie_ids, relevant, 10),
            "NDCG@10_Fuzzy": ndcg_at_k(top_movie_ids, relevant, 10),
        })

    df = pd.DataFrame(results)
    means = df[["Precision@10_Fuzzy", "Recall@10_Fuzzy", "NDCG@10_Fuzzy"]].mean(numeric_only=True)
    print("Average metrics on cold-start users:")
    for k, v in means.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

