import pandas as pd

def load_movielens_1m(path_prefix: str = "ml-1m"):
    ratings = pd.read_csv(
        f"{path_prefix}/ratings.dat",
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )

    users = pd.read_csv(
        f"{path_prefix}/users.dat",
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "ZipCode"],
    )
    return ratings, users

ratings, users = load_movielens_1m()
ratings.head(), users.head()
