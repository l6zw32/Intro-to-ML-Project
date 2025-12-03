import pandas as pd
from typing import Dict, List

def _build_group_key(df: pd.DataFrame) -> pd.Series:

    return (
        df["Gender"].astype(str)
        + "_"
        + df["Age"].astype(str)
        + "_"
        + df["Occupation"].astype(str)
    )


def recommend_demographic_popularity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    users_df: pd.DataFrame,
    k: int = 10,
) -> Dict[int, List[int]]:

    # Merge ratings with user metadata
    merged = pd.merge(train_df, users_df, on="UserID", how="left")
    merged["GroupKey"] = _build_group_key(merged)

    # Popularity per (GroupKey, MovieID)
    group_item_pop = (
        merged.groupby(["GroupKey", "MovieID"])["Rating"]
        .count()
        .reset_index(name="Count")
    )

    # Precompute sorted movie lists per group
    group_to_items: Dict[str, List[int]] = {}
    for g, sub in group_item_pop.groupby("GroupKey"):
        sub_sorted = sub.sort_values("Count", ascending=False)
        group_to_items[g] = sub_sorted["MovieID"].tolist()

    # Global popularity as fallback
    global_pop = (
        train_df.groupby("MovieID")["Rating"]
        .count()
        .sort_values(ascending=False)
        .index.to_list()
    )

    user_meta = users_df.set_index("UserID")
    user_seen = train_df.groupby("UserID")["MovieID"].apply(set).to_dict()
    target_users = test_df["UserID"].unique().tolist()

    recs: Dict[int, List[int]] = {}

    for uid in target_users:
        seen = user_seen.get(uid, set())

        if uid in user_meta.index:
            row = user_meta.loc[uid]
            gkey = f"{row['Gender']}_{row['Age']}_{row['Occupation']}"
            group_items = group_to_items.get(gkey, global_pop)
        else:
            group_items = global_pop

        candidates = [iid for iid in group_items if iid not in seen]
        recs[uid] = candidates[:k]

    return recs