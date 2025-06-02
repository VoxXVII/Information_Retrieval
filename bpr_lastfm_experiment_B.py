import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from implicit.bpr import BayesianPersonalizedRanking
import matplotlib.pyplot as plt
from pathlib import Path

# --- Evaluation Metrics ---
def precision_at_k(model, train_user_items, test_user_items, K=10):
    precisions = []
    for user in range(train_user_items.shape[0]):
        test_items = test_user_items[user].indices
        if len(test_items) == 0:
            continue
        user_items = train_user_items[user]
        recommended = [itemid for itemid, *_ in model.recommend(user, user_items, N=K, filter_already_liked_items=True)]
        hits = len(set(recommended) & set(test_items))
        precisions.append(hits / K)
    return np.mean(precisions) if precisions else 0.0

def recall_at_k(model, train_user_items, test_user_items, K=10):
    recalls = []
    for user in range(train_user_items.shape[0]):
        test_items = test_user_items[user].indices
        if len(test_items) == 0:
            continue
        user_items = train_user_items[user]
        recommended = [itemid for itemid, *_ in model.recommend(user, user_items, N=K, filter_already_liked_items=True)]
        hits = len(set(recommended) & set(test_items))
        recalls.append(hits / len(test_items))
    return np.mean(recalls) if recalls else 0.0

# --- Load Last.fm data ---
file_path = Path("C:/Users/vasil/OneDrive/Έγγραφα/Information_Retrieval/user_artists.dat")
df = pd.read_csv(file_path, sep='\t', names=["userID", "artistID", "weight"], header=0)

print("Loaded user-artist interactions:")
print(df.head())

# --- Map user & artist IDs to 0-based indices ---
user_ids = df['userID'].unique()
artist_ids = df['artistID'].unique()

user_id_map = {old: new for new, old in enumerate(user_ids)}
artist_id_map = {old: new for new, old in enumerate(artist_ids)}

df['user_idx'] = df['userID'].map(user_id_map)
df['artist_idx'] = df['artistID'].map(artist_id_map)

num_users = len(user_id_map)
num_artists = len(artist_id_map)

# --- Build interaction matrix ---
full_matrix = sparse.lil_matrix((num_users, num_artists))
for row in df.itertuples():
    full_matrix[row.user_idx, row.artist_idx] = 1.0  # binary implicit feedback
full_matrix = full_matrix.tocsr()

# --- Train/test split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_matrix = sparse.lil_matrix((num_users, num_artists))
for row in train_df.itertuples():
    train_matrix[row.user_idx, row.artist_idx] = 1.0
train_matrix = train_matrix.tocsr()

test_matrix = sparse.lil_matrix((num_users, num_artists))
for row in test_df.itertuples():
    test_matrix[row.user_idx, row.artist_idx] = 1.0
test_matrix = test_matrix.tocsr()

# --- Model Training ---
factors = 50
k_values = list(range(2, 22, 2))
precisions, recalls = [], []

print(f"Training BPR with {factors} latent factors for varying top-K...")

model = BayesianPersonalizedRanking(factors=factors, iterations=20)
model.fit(train_matrix)

# --- Evaluation ---
for k in k_values:
    precision = precision_at_k(model, train_matrix, test_matrix, K=k)
    recall = recall_at_k(model, train_matrix, test_matrix, K=k)
    precisions.append(precision)
    recalls.append(recall)
    print(f"K={k} | Precision@{k}={precision:.4f} | Recall@{k}={recall:.4f}")

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.plot(k_values, precisions, marker='o', label='Precision@K')
plt.title("Precision@K vs K (Last.fm, 50 Latent Factors)")
plt.xlabel("K (Top-K Recommendations)")
plt.ylabel("Precision@K")
plt.grid(True)
plt.legend()
plt.savefig("precision_vs_k_lastfm.png")

plt.figure(figsize=(10, 5))
plt.plot(k_values, recalls, marker='s', color='orange', label='Recall@K')
plt.title("Recall@K vs K (Last.fm, 50 Latent Factors)")
plt.xlabel("K (Top-K Recommendations)")
plt.ylabel("Recall@K")
plt.grid(True)
plt.legend()
plt.savefig("recall_vs_k_lastfm.png")
