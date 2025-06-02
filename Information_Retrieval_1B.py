import networkx as nx
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

# Create the undirected graph
G = nx.DiGraph()
edges = [
   [0,1],[0,6],[0,7],
   [1,2],[1,7],
   [2,1],[2,7],
   [3,5],[3,7],
   [4,5],
   [5,6],
   [6,5],
   [7,6]
]
G.add_edges_from(edges)

# --- PageRank computations ---
damping_factors = [0.55, 0.65, 0.75, 0.85, 0.95]
pagerank_results = {}

for alpha in damping_factors:
    pr = nx.pagerank(G, alpha=alpha, max_iter=1000, tol=1.0e-6)
    pagerank_results[alpha] = pr

# --- Plot PageRank variation per node ---
plt.figure(figsize=(10, 6))
nodes = list(G.nodes())
colors = sns.color_palette("tab10", len(nodes))

# Small vertical offset to separate flat lines slightly
offsets = {node: i * 0.0005 for i, node in enumerate(nodes)}

for i, node in enumerate(nodes):
    y_vals = [pagerank_results[a][node] + offsets[node] for a in damping_factors]
    plt.plot(
        damping_factors,
        y_vals,
        marker='o',
        label=f'Node {node}',
        color=colors[i],
        linewidth=2,
    )
    for x, y in zip(damping_factors, y_vals):
        plt.text(x, y + 0.0002, str(node), fontsize=8, ha='center', color=colors[i])

plt.title("PageRank vs Damping Factor for Each Node")
plt.xlabel("Damping Factor (α)")
plt.ylabel("PageRank Value (with slight offsets for visibility)")
plt.legend(ncol=2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# --- Kendall's Tau between ranking orders ---
rankings = {}

for alpha in damping_factors:
    sorted_nodes = sorted(pagerank_results[alpha].items(), key=lambda x: x[1], reverse=True)
    rankings[alpha] = [node for node, _ in sorted_nodes]

print("\nKendall's Tau between different damping factors:\n")
for a1, a2 in itertools.combinations(damping_factors, 2):
    tau, p_value = kendalltau(rankings[a1], rankings[a2])
    print(f"α = {a1:.2f} vs α = {a2:.2f} ➜ τ = {tau:.4f}, p = {p_value:.4f}")