import networkx as nx

# Create the undirected graph
G = nx.Graph()
edges = [
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
G.add_edges_from(edges)

# Define personalization vector
personalization = {}
nodes = list(G.nodes())
target_node = 14
other_nodes = [n for n in nodes if n != target_node]
N = len(nodes)

for node in nodes:
    if node == target_node:
        personalization[node] = 0.5
    else:
        personalization[node] = 0.5 / (N - 1)

# Compute PageRank with damping factor alpha = 0.65
pagerank = nx.pagerank(G, alpha=0.65, personalization=personalization)

# Display sorted PageRank values
for node, pr in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
    print(f"Node {node}: {pr:.4f}")
