import pandas as pd
from typing import Dict, List

def recommend_global_popularity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10, # 10 recommendations per user
) -> Dict[int, List[int]]:

    # Global popularity: count of ratings per movie
    pop = (
        train_df.groupby("MovieID")["Rating"]
        .count()
        .sort_values(ascending=False)
    )
    popular_items = pop.index.to_list()

    # Items each user has already seen in train
    user_seen = (
        train_df.groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )

    target_users = test_df["UserID"].unique().tolist()
    recs: Dict[int, List[int]] = {}

    for uid in target_users:
        seen = user_seen.get(uid, set())
        candidates = [iid for iid in popular_items if iid not in seen]
        recs[uid] = candidates[:k]

    return recs