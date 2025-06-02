import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.linalg import eig

def build_transition_matrix(edges, num_nodes):
    adjacency = np.zeros((num_nodes, num_nodes))
    for src, dst in edges:
        adjacency[dst - 1][src - 1] = 1  # Transpose: column is source, row is destination

    # Normalize columns (each column sums to 1 for stochasticity)
    column_sums = adjacency.sum(axis=0)
    for i in range(num_nodes):
        if column_sums[i] != 0:
            adjacency[:, i] /= column_sums[i]
        else:
            # If no outlinks, make it a dangling node that links to everyone
            adjacency[:, i] = 1 / num_nodes

    return adjacency

def build_google_matrix(transition_matrix, alpha):
    n = transition_matrix.shape[0]
    return alpha * transition_matrix + (1 - alpha) * np.ones((n, n)) / n


def compute_all_eigenvalues(matrix):
    eigenvalues, _ = eig(matrix)
    return np.sort(np.real(eigenvalues))[::-1]


def main():
    alpha = 0.85
    num_nodes = 14
    original_edges = [
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 1), (2, 3), (2, 5),
        (3, 1), (3, 2), (3, 4),
        (4, 1), (4, 3), (4, 5),
        (5, 1), (5, 2), (5, 4), (5, 10),
        (6, 3), (6, 7), (6, 8), (6, 9),
        (7, 4), (7, 6), (7, 9),
        (8, 6), (8, 9),
        (9, 6), (9, 7), (9, 8),
        (10, 5), (10, 11), (10, 12), (10, 14),
        (11, 10), (11, 13), (11, 14),
        (12, 10), (12, 13), (12, 14),
        (13, 11), (13, 12), (13, 14),
        (14, 10), (14, 11), (14, 12), (14, 13)
    ]

    # Compute eigenvalues before modification
    trans_matrix_orig = build_transition_matrix(original_edges, num_nodes)
    google_matrix_orig = build_google_matrix(trans_matrix_orig, alpha)
    eigenvalues_orig = compute_all_eigenvalues(google_matrix_orig)
    print("Eigenvalues before adding edges:\n", eigenvalues_orig)

    # Add new edges
    new_edges = original_edges + [(5, 11), (4, 11), (7, 13), (8, 12), (9, 13)]
    trans_matrix_new = build_transition_matrix(new_edges, num_nodes)
    google_matrix_new = build_google_matrix(trans_matrix_new, alpha)
    eigenvalues_new = compute_all_eigenvalues(google_matrix_new)
    print("\nEigenvalues after adding edges:\n", eigenvalues_new)

if __name__ == "__main__":
    main()
