import networkx as nx
import numpy as np

# Define the directed graph based on the image
edges = [
    [3,8], [3,10],
    [5,11],
    [7,8], [7,11],
    [8,9],
    [11,2], [11,9], [11,10]
]

G = nx.DiGraph()
G.add_edges_from(edges)

# Create the adjacency matrix A
A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
print("Adjacency matrix (A):\n", A)

# Compute Authority matrix: AᵀA
authority_matrix = np.dot(A.T, A)
print("\nAuthority Matrix (AᵀA):\n", authority_matrix)

# Compute Hub matrix: AAᵀ
hub_matrix = np.dot(A, A.T)
print("\nHub Matrix (AAᵀ):\n", hub_matrix)

# Compute eigenvectors
eigvals_auth, eigvecs_auth = np.linalg.eig(authority_matrix)
eigvals_hub, eigvecs_hub = np.linalg.eig(hub_matrix)

# Get the principal eigenvector (the one with the largest eigenvalue)
principal_eigvec_auth = eigvecs_auth[:, np.argmax(np.real(eigvals_auth))]
principal_eigvec_hub = eigvecs_hub[:, np.argmax(np.real(eigvals_hub))]

print("\nPrincipal Eigenvector (Authorities):\n", principal_eigvec_auth)
print("\nPrincipal Eigenvector (Hubs):\n", principal_eigvec_hub)
