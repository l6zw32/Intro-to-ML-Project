import pandas as pd
from typing import Dict, List, Set
import numpy as np
from collections import defaultdict

def precision_at_k(predicted: List[int], relevant: Set[int], k: int) -> float:
    predicted_k = predicted[:k]
    if k == 0:
        return 0.0
    return len(set(predicted_k) & relevant) / float(k) 

def recall_at_k(predicted: List[int], relevant: Set[int], k: int) -> float:
    predicted_k = predicted[:k]
    if not relevant:
        return 0.0
    return len(set(predicted_k) & relevant) / float(len(relevant))

def ndcg_at_k(predicted: List[int], relevant: Set[int], k: int) -> float:
    predicted_k = predicted[:k]
    dcg = 0.0
    for i, iid in enumerate(predicted_k):
        if iid in relevant:
            dcg += 1.0 / np.log2(i + 2.0)  # rank starts at 1
    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0
    
def evaluate_recommender(
    recs: Dict[int, List[int]],
    test_df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:

    ground_truth: Dict[int, Set[int]] = defaultdict(set)
    for row in test_df.itertuples(index=False):
        ground_truth[row.UserID].add(row.MovieID)

    results = []
    for uid, relevant in ground_truth.items():
        predicted = recs.get(uid, [])
        results.append(
            {
                "UserID": uid,
                f"Precision@{k}": precision_at_k(predicted, relevant, k),
                f"Recall@{k}": recall_at_k(predicted, relevant, k),
                f"NDCG@{k}": ndcg_at_k(predicted, relevant, k),
            }
        )
    return pd.DataFrame(results)