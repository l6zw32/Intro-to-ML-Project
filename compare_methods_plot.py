import numpy as np
import matplotlib.pyplot as plt
from load_data import users
from make_cold_start_split import test_df, train_df
from global_popularity import recommend_global_popularity
from demographics_based_popularity import recommend_demographic_popularity
from eval_methods import evaluate_recommender

# recommendations for cold-start users using global popularity
recs_global = recommend_global_popularity(train_df, test_df, k=10)

# Evaluate global popularity recommender
eval_global = evaluate_recommender(recs_global, test_df, k=10)
eval_global.head()

# Get demographic-popularity recommendations
recs_demo = recommend_demographic_popularity(train_df, test_df, users, k=10)

# Evaluate the demographic-popularity recommender
eval_demo = evaluate_recommender(recs_demo, test_df, k=10)

# Show first few rows and mean metrics
print(eval_demo.head())
print("Mean metrics (Demographic Popularity):")
print(eval_demo.mean())

mean_global = eval_global[["Precision@10", "Recall@10", "NDCG@10"]].mean()
mean_demo   = eval_demo[["Precision@10", "Recall@10", "NDCG@10"]].mean()

metrics = ["Precision@10", "Recall@10", "NDCG@10"]
methods = ["Global Popularity", "Demographic Popularity"] # we can add more methods here as they are ready

values_global = [mean_global[m] for m in metrics]
values_demo   = [mean_demo[m]   for m in metrics]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, values_global, width, label=methods[0])
ax.bar(x + width/2, values_demo,   width, label=methods[1])

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
plt.savefig("Mean @10 Metrics for Cold-Start Users.png")
plt.show()

ndcg_global = eval_global["NDCG@10"].values
ndcg_demo   = eval_demo["NDCG@10"].values
data   = [ndcg_global, ndcg_demo]
labels = ["Global Popularity", "Demographic Popularity"]
fig, ax = plt.subplots()
ax.boxplot(data, labels=labels, showfliers=False)
ax.set_ylabel("NDCG@10")
plt.savefig("Distribution of NDCG@10 for Cold-Start Users.png")
plt.show()

