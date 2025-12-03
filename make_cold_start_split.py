import numpy as np
import random
import pandas as pd
from load_data import ratings

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def make_cold_start_split(
    ratings: pd.DataFrame,
    min_known: int = 5,
    n_cold_users: int = 500,
):

    user_counts = ratings["UserID"].value_counts()
    eligible_users = user_counts[user_counts >= min_known + 1].index.tolist() #only users that have atleast 6 ratings are considered

    if n_cold_users is not None and n_cold_users < len(eligible_users):
        cold_users = random.sample(eligible_users, n_cold_users)
    else:
        cold_users = eligible_users # select 500 cold start users out of the eligible ones

    train_rows = []
    test_rows = []

    for uid, user_ratings in ratings.groupby("UserID"):
        if uid in cold_users:
            # shuffle for randomness
            user_ratings = user_ratings.sample(frac=1.0, random_state=RANDOM_SEED)
            known = user_ratings.iloc[:min_known]
            held_out = user_ratings.iloc[min_known:]
            train_rows.append(known)
            test_rows.append(held_out)
        else:
            train_rows.append(user_ratings)

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    return train_df, test_df

train_df, test_df = make_cold_start_split(ratings, min_known=5, n_cold_users=500) #train and test the data on cold start users
train_df.head(), test_df.head()