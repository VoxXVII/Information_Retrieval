import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from implicit.bpr import BayesianPersonalizedRanking
import matplotlib.pyplot as plt
import os

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

# Load user-artist implicit feedback dataset (listening counts)
df = pd.read_csv("/home/aris/Downloads/hetrec2011-lastfm-2k/user_artists.dat", sep='\t')
print("Loaded user-artist interactions:")
print(df.head())

# Map original IDs to 0-based indices for efficient matrix use
user_ids = df['userID'].unique()
artist_ids = df['artistID'].unique()
user_id_map = {old: new for new, old in enumerate(user_ids)}
artist_id_map = {old: new for new, old in enumerate(artist_ids)}

# Create new columns with mapped indices
df['user_idx'] = df['userID'].map(user_id_map)
df['artist_idx'] = df['artistID'].map(artist_id_map)

# Number of users and artists
num_users = len(user_id_map)
num_artists = len(artist_id_map)

# Build full interaction matrix with listening counts
full_matrix = sparse.lil_matrix((num_users, num_artists))
for row in df.itertuples():
    full_matrix[row.user_idx, row.artist_idx] = 1.0  # implicit feedback: any listen is positive
full_matrix = full_matrix.tocsr()

# Train/test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Train matrix
train_matrix = sparse.lil_matrix((num_users, num_artists))
for row in train_data.itertuples():
    train_matrix[row.user_idx, row.artist_idx] = 1.0
train_matrix = train_matrix.tocsr()

# Test matrix
test_matrix = sparse.lil_matrix((num_users, num_artists))
for row in test_data.itertuples():
    test_matrix[row.user_idx, row.artist_idx] = 1.0
test_matrix = test_matrix.tocsr()

# BPR Hyperparameter Experimentation: latent factors from 10 to 100
factors_range = list(range(10, 110, 10))
precisions_all_runs = []
recalls_all_runs = []

num_runs = 10

for run in range(num_runs):
    print(f"Run {run+1}/{num_runs}")
    precisions = []
    recalls = []
    for factors in factors_range:
        print(f"  Training BPR with {factors} latent factors...")
        model = BayesianPersonalizedRanking(factors=factors, iterations=20)
        model.fit(train_matrix)

        precision = precision_at_k(model, train_matrix, test_matrix, K=10)
        recall = recall_at_k(model, train_matrix, test_matrix, K=10)

        precisions.append(precision)
        recalls.append(recall)

        print(f"  ✓ Factors={factors} | Precision@10={precision:.4f} | Recall@10={recall:.4f}")

    precisions_all_runs.append(precisions)
    recalls_all_runs.append(recalls)

# Convert to numpy arrays
precisions_all_runs = np.array(precisions_all_runs)
recalls_all_runs = np.array(recalls_all_runs)

# Mean and std for each factor
precisions_mean = precisions_all_runs.mean(axis=0)
precisions_std = precisions_all_runs.std(axis=0)
recalls_mean = recalls_all_runs.mean(axis=0)
recalls_std = recalls_all_runs.std(axis=0)

# Plot Precision@10
plt.figure(figsize=(10, 5))
plt.plot(factors_range, precisions_mean, marker='o', label='Precision@10 Mean')
plt.fill_between(factors_range, precisions_mean - precisions_std, precisions_mean + precisions_std, alpha=0.2)
plt.title("Precision@10 vs Latent Factors (mean ± std)")
plt.xlabel("Latent Factors")
plt.ylabel("Precision@10")
plt.grid(True)
plt.legend()
plt.savefig("precision_latent_factors_A.png")

# Plot Recall@10
plt.figure(figsize=(10, 5))
plt.plot(factors_range, recalls_mean, marker='s', color='orange', label='Recall@10 Mean')
plt.fill_between(factors_range, recalls_mean - recalls_std, recalls_mean + recalls_std, alpha=0.2, color='orange')
plt.title("Recall@10 vs Latent Factors (mean ± std)")
plt.xlabel("Latent Factors")
plt.ylabel("Recall@10")
plt.grid(True)
plt.legend()
plt.savefig("recall_latent_factors_A.png")