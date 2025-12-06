from surprise import AlgoBase
from collections import defaultdict
import numpy as np

class GlobalPopularity(AlgoBase):
    """
    Popularity-based recommender compatible with Surprise.
    Scores items by how often they were rated in the trainset.
    """

    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        item_counts = defaultdict(int)
        for uid, iid, r in trainset.all_ratings():
            item_counts[iid] += 1

        self.item_counts = dict(item_counts)
        self.max_count = max(self.item_counts.values()) if self.item_counts else 1

        return self

    def estimate(self, uid, iid):
        # If item unseen, use global mean
        if iid not in self.item_counts:
            return self.trainset.global_mean

        count = self.item_counts[iid]
        # Map popularity to [1,5]
        return 1.0 + 4.0 * (count / self.max_count)
