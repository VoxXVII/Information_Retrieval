import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from implicit.bpr import BayesianPersonalizedRanking
import matplotlib.pyplot as plt
from pathlib import Path

# Precision@K
def precision_at_k(model, train_user_items, test_user_items, K=10):
    precisions = []
    num_users = train_user_items.shape[0]
    for user in range(num_users):
        test_items = test_user_items[user].indices
        if len(test_items) == 0:
            continue
        user_items = train_user_items[user]
        recommended = [itemid for itemid, *_ in model.recommend(user, user_items, N=K, filter_already_liked_items=True)]
        hits = len(set(recommended) & set(test_items))
        precisions.append(hits / K)
    return np.mean(precisions) if precisions else 0.0

# Recall@K
def recall_at_k(model, train_user_items, test_user_items, K=10):
    recalls = []
    num_users = train_user_items.shape[0]
    for user in range(num_users):
        test_items = test_user_items[user].indices
        if len(test_items) == 0:
            continue
        user_items = train_user_items[user]
        recommended = [itemid for itemid, *_ in model.recommend(user, user_items, N=K, filter_already_liked_items=True)]
        hits = len(set(recommended) & set(test_items))
        recalls.append(hits / len(test_items))
    return np.mean(recalls) if recalls else 0.0

# Load MovieLens 1M dataset (implicit feedback: any rating counts as interaction)
file_path = Path("C:/Users/vasil/OneDrive/Έγγραφα/Information_Retrieval/ml-1m/ratings.dat")
df = pd.read_csv(file_path, sep='::', engine='python', names=['userID', 'movieID', 'rating', 'timestamp'])

print("Loaded user-movie interactions:")
print(df.head())

# Map to 0-based indices for matrix use
user_ids = df['userID'].unique()
movie_ids = df['movieID'].unique()
user_id_map = {old: new for new, old in enumerate(user_ids)}
movie_id_map = {old: new for new, old in enumerate(movie_ids)}

df['user_idx'] = df['userID'].map(user_id_map)
df['movie_idx'] = df['movieID'].map(movie_id_map)

num_users = len(user_id_map)
num_movies = len(movie_id_map)

# Build interaction matrix (1.0 for any rating, since it's implicit feedback)
full_matrix = sparse.lil_matrix((num_users, num_movies))
for row in df.itertuples():
    full_matrix[row.user_idx, row.movie_idx] = 1.0
full_matrix = full_matrix.tocsr()

# Train/test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Train matrix
train_matrix = sparse.lil_matrix((num_users, num_movies))
for row in train_data.itertuples():
    train_matrix[row.user_idx, row.movie_idx] = 1.0
train_matrix = train_matrix.tocsr()

# Test matrix
test_matrix = sparse.lil_matrix((num_users, num_movies))
for row in test_data.itertuples():
    test_matrix[row.user_idx, row.movie_idx] = 1.0
test_matrix = test_matrix.tocsr()

# Fixed latent factor
factors = 50
k_values = list(range(2, 22, 2))  # K from 2 to 20 step 2
num_runs = 10

precisions_k_all_runs = []
recalls_k_all_runs = []

for run in range(num_runs):
    print(f"Experiment B - Run {run+1}/{num_runs}")
    model = BayesianPersonalizedRanking(factors=factors, iterations=20)
    model.fit(train_matrix)

    precisions_k = []
    recalls_k = []
    for k in k_values:
        precision = precision_at_k(model, train_matrix, test_matrix, K=k)
        recall = recall_at_k(model, train_matrix, test_matrix, K=k)
        precisions_k.append(precision)
        recalls_k.append(recall)
        print(f"K={k} | Precision@{k}={precision:.4f} | Recall@{k}={recall:.4f}")

    precisions_k_all_runs.append(precisions_k)
    recalls_k_all_runs.append(recalls_k)

precisions_k_all_runs = np.array(precisions_k_all_runs)
recalls_k_all_runs = np.array(recalls_k_all_runs)

precisions_k_mean = precisions_k_all_runs.mean(axis=0)
precisions_k_std = precisions_k_all_runs.std(axis=0)
recalls_k_mean = recalls_k_all_runs.mean(axis=0)
recalls_k_std = recalls_k_all_runs.std(axis=0)

# Plot Precision@K
plt.figure(figsize=(10, 5))
plt.plot(k_values, precisions_k_mean, marker='o', label='Precision@K Mean')
plt.fill_between(k_values, precisions_k_mean - precisions_k_std, precisions_k_mean + precisions_k_std, alpha=0.2)
plt.title("Precision@K vs K (mean ± std), factors=50")
plt.xlabel("K (Top-K Recommendations)")
plt.ylabel("Precision@K")
plt.grid(True)
plt.legend()
plt.savefig("precision_vs_k_movielens.png")

# Plot Recall@K
plt.figure(figsize=(10, 5))
plt.plot(k_values, recalls_k_mean, marker='s', color='orange', label='Recall@K Mean')
plt.fill_between(k_values, recalls_k_mean - recalls_k_std, recalls_k_mean + recalls_k_std, alpha=0.2, color='orange')
plt.title("Recall@K vs K (mean ± std), factors=50")
plt.xlabel("K (Top-K Recommendations)")
plt.ylabel("Recall@K")
plt.grid(True)
plt.legend()
plt.savefig("recall_vs_k_movielens.png")