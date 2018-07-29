import community


def calculate_modularity(graph, clusters):
    G = graph.to_undirected()
    parts = dict(zip(G.nodes(), clusters))
    return community.modularity(parts, G)
