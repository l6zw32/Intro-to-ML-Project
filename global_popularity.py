import pandas as pd
from typing import Dict, List
from fuzzy import *

def recommend_global_popularity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10, # 10 recommendations per user
) -> Dict[int, List[Tuple[int, float]]]:

    # Global popularity: count of ratings per movie
    pop = (
        train_df.groupby("MovieID")["Rating"]
        .count()
        .sort_values(ascending=False)
    )
    max_count = pop.iloc[0]
    popular_items = pop.index.to_list()

    # Items each user has already seen in train
    user_seen = (
        train_df.groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )

    target_users = test_df["UserID"].unique().tolist()
    recs: Dict[int, List[Tuple[int, float]]] = {}

    for uid in target_users:
        seen = user_seen.get(uid, set())
        candidates = [(iid, 5.0 * pop[iid] / max_count) for iid in popular_items if iid not in seen]
        recs[uid] = candidates[:k]

    return recs

if __name__ == "__main__":
    ratings, users, movies = load_movielens_dfs()
    train_df, test_df = split_cold_start(ratings, seed=FUZZY_RANDOM_SEED, cold_frac=0.2, revealed_per_user=5)
    recs = recommend_global_popularity(train_df, test_df, k=10)
    print(recs[0])
