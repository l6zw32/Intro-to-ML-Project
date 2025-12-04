from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from load_data import users
from make_cold_start_split import test_df, train_df
from global_popularity import recommend_global_popularity
from demographics_based_popularity import recommend_demographic_popularity

from rdfsvd import RDFSVD
from utils import cold_start_train, evaluate

from fuzzy import *

from surprise import Dataset, Reader

def convert_to_surprise_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, rating_scale=(1, 5)):
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(train_df[["UserID", "MovieID", "Rating"]], reader)
    train = data.build_full_trainset()
    test_data = Dataset.load_from_df(test_df[["UserID", "MovieID", "Rating"]], reader)
    test = test_data.build_full_trainset().build_testset()
    return train, test


def get_rdfsvd_scores(predictions):
    scores = defaultdict(list)  # {user: [(item, score)]}
    for pred in predictions:
        if not pred.details.get("was_impossible", False):
            scores[int(pred.uid)].append((int(pred.iid), float(pred.est)))
    return scores  # Dict[int, List[Tuple[int, float]]]

def normalize_user_scores(user_scores: List[Tuple[int, float]]) -> Dict[int, float]:
    if not user_scores:
        return {}
    scores = [s for _, s in user_scores]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return {mid: 0.0 for mid, _ in user_scores}
    return {mid: (s - min_s) / (max_s - min_s) for mid, s in user_scores}

def ensemble_recommendations(
    rdf_recs: Dict[int, List[Tuple[int, float]]],
    fuzzy_recs: Dict[int, List[Tuple[int, float]]],
    pop_recs: Dict[int, List[Tuple[int, float]]],
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),  # RDF, Fuzzy, Pop
    top_n: int = 10,
) -> Dict[int, List[int]]:

    final_recs = {}
    all_users = set(rdf_recs.keys()) | set(fuzzy_recs.keys()) | set(pop_recs.keys())

    for uid in all_users:
        score_dict = defaultdict(float)
        
        for iid, score in rdf_recs.get(uid, []):
            score_dict[iid] += weights[0] * score
        for iid, score in fuzzy_recs.get(uid, []):
            score_dict[iid] += weights[1] * score
        for iid, score in pop_recs.get(uid, []):
            score_dict[iid] += weights[2] * score

        ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        final_recs[uid] = ranked[:top_n]

    return final_recs

def load_recs_from_files(filenames: List[str]) -> List[Dict[int, List[Tuple[int, float]]]]:
    recs_list = []
    for filename in filenames:
        recs = {}
        with open(filename, "r") as f:
            for line in f:
                uid_str, items_str = line.strip().split(":")
                uid = int(uid_str)
                items = eval(items_str.strip())
                recs[uid] = items
        recs_list.append(recs)
    return recs_list

def precision_at_k(recs: Dict[int, List[int]], test_df: pd.DataFrame, k: int = 10) -> float:
    test_truth = (
        test_df.groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )
    precisions = []
    for uid, pred_items in recs.items():
        if uid not in test_truth:
            continue
        true_items = test_truth[uid]
        pred_topk = [iid for iid, _ in pred_items[:k]]
        num_hit = len(set(pred_topk) & true_items)
        precisions.append(num_hit / k)
    return sum(precisions) / len(precisions)

def recall_at_k(recs: Dict[int, List[int]], test_df: pd.DataFrame, k: int = 10) -> float:
    test_truth = (
        test_df.groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )
    recalls = []
    for uid, pred_items in recs.items():
        if uid not in test_truth:
            continue
        true_items = test_truth[uid]
        pred_topk = [iid for iid, _ in pred_items[:k]]
        num_hit = len(set(pred_topk) & true_items)
        recalls.append(num_hit / len(true_items) if true_items else 0)
    return sum(recalls) / len(recalls)

def ndcg_at_k(recs: Dict[int, List[Tuple[int, float]]], test_df: pd.DataFrame, k: int = 10) -> float:
    test_truth = (
        test_df.groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )
    def dcg(relevances):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))

    ndcgs = []
    for uid, pred_items in recs.items():
        if uid not in test_truth:
            continue
        true_items = test_truth[uid]
        pred_topk = [iid for iid, _ in pred_items[:k]]
        relevances = [1 if iid in true_items else 0 for iid in pred_topk]
        ideal_relevances = sorted(relevances, reverse=True)
        score = dcg(relevances) / (dcg(ideal_relevances) + 1e-12)
        ndcgs.append(score)
    return sum(ndcgs) / len(ndcgs)


def train_and_rec():
    ratings, users, movies = load_movielens_dfs()
    train_df, test_df = split_cold_start(ratings, seed=FUZZY_RANDOM_SEED, cold_frac=0.2, revealed_per_user=5)
    surprise_trainset, surprise_testset = convert_to_surprise_dataset(train_df, test_df)

    pop_scores = recommend_global_popularity(train_df, test_df, k=10)
    fuzzy_model = train_fuzzy_user_model(train_df)
    fuzzy_scores = fuzzy_predict_ratings(fuzzy_model, ratings, train_df, test_df)


    rdfsvd_alg = RDFSVD(surprise_trainset.n_users, surprise_trainset.n_items, n_epochs=10)
    print("Fitting RDFSVD...")
    rdfsvd_alg.fit(surprise_trainset)
    print("Predicting with RDFSVD...")
    predictions = rdfsvd_alg.test(surprise_testset)
    rdfsvd_scores = get_rdfsvd_scores(predictions)
    
    with open("pop_scores.txt", "w") as f:
        for uid, items in pop_scores.items():
            f.write(f"{uid}: {items}\n")
    with open("fuzzy_scores.txt", "w") as f:
        for uid, items in fuzzy_scores.items():
            f.write(f"{uid}: {items}\n")
    with open("rdfsvd_scores.txt", "w") as f:
        for uid, items in rdfsvd_scores.items():
            f.write(f"{uid}: {items}\n")

def main():
    # train_and_rec()
    rdfsvd_recs, fuzzy_recs, pop_recs = load_recs_from_files(
        ["rdfsvd_scores.txt", "fuzzy_scores.txt", "pop_scores.txt"]
    )
    final_recs = ensemble_recommendations(
        rdfsvd_recs, fuzzy_recs, pop_recs, weights=(0.7, 0.2, 0.1), top_n=10
    )
    
    for rec in [rdfsvd_recs, fuzzy_recs, pop_recs]:
        print("Precision@10:", precision_at_k(rec, test_df))
        print("Recall@10:", recall_at_k(rec, test_df))
        print("NDCG@10:", ndcg_at_k(rec, test_df))
        print("-----")

    print("Ensembled Results:")
    print("Precision@10:", precision_at_k(final_recs, test_df))
    print("Recall@10:", recall_at_k(final_recs, test_df))
    print("NDCG@10:", ndcg_at_k(final_recs, test_df))


if __name__ == "__main__":
    main()
