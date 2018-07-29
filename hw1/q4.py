import os
import matplotlib
import networkx as nx
import networkx.algorithms as algos
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from data import data
import numpy as np
import community

snap_available = True
try:
    import snap
except:
    snap_available = False
    print "Snap isn't available, fallback to networkx"

gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
for gui in gui_env:
    try:
        print "Testing matplotlib backend...", gui
        matplotlib.use(gui, warn=False, force=True)
        matplotlib.interactive(False)
        from matplotlib import pyplot as plt

        break
    except Exception as e:
        continue

if __name__ == '__main__':
    E = data.load_wiki_vote()
    # E = np.array([[1, 2], [2, 3], [3, 6], [6, 1], [7, 1]]) # for test
    report = []
    # calculate measures
    G = nx.Graph()
    G.add_edges_from(E)

    # Betweenness coefficient
    if snap_available:
        G_snap = snap.LoadEdgeList(snap.TNGraph, "data/Datasets/Wiki-Vote.txt", 0, 1)
        average_clustering_coef = snap.GetClustCf(G_snap, -1)
        report += ["Average clustering coefficient: %f" % float(average_clustering_coef)]
        print report[-1]
        V = snap.TIntFltH()
        E = snap.TIntPrFltH()
        snap.GetBetweennessCentr(G_snap, V, E, 1.0)
        values = []
        for i in V:
            values.append(V[i])
        n = len(values)
        betweenness_coef = sum(values) / ((n - 1) * (n - 2))
        report += ["Average betweenness coefficient: %f" % float(betweenness_coef)]
        print report[-1]
    else:
        # Clustering coefficient
        average_clustering_coef = nx.average_clustering(G)
        report += ["Average clustering coefficient: %f" % float(average_clustering_coef)]
        print report[-1]
        values = algos.betweenness_centrality(G)
        n = len(values)
        betweenness_coef = sum(values) / ((n - 1) * (n - 2))
        report += ["Average betweenness coefficient: %f" % float(betweenness_coef)]
        print report[-1]
    # Saving eigs for faster computations
    if not (os.path.exists("outputs/q4/intermediates/vecs.npy") and
                os.path.exists("outputs/q4/intermediates/vals.npy")):
        laplacian_matrix = nx.linalg.laplacian_matrix(G).todense()
        vals, vecs = np.linalg.eigh(laplacian_matrix)
        np.save('outputs/q4/intermediates/vecs.npy', vecs)
        np.save('outputs/q4/intermediates/vals.npy', vals)
    else:
        vecs = np.load('outputs/q4/intermediates/vecs.npy')
        vals = np.load('outputs/q4/intermediates/vals.npy')

    print "Run KMeans clustering on laplacian matrix..."
    k = 30
    report += ["KMeans using k=%d --> outputs/q4/idx.csv" % k]
    some_vecs = vecs[:, :k]
    clusters = KMeans(n_clusters=k).fit(scale(some_vecs))
    print "KMeans clustering done!"
    np.savetxt('outputs/q4/idx.csv', clusters.labels_, delimiter=",", fmt="%d")
    print "Saving calculated labels to outputs/q4/idx.csv!"
    parts = dict(zip(G.nodes(), clusters.labels_))
    modularity = community.modularity(parts, G)
    report += ['Modularity: %f' % modularity]
    print report[-1]
    report += ['Minimum cut: %d' % len(nx.minimum_edge_cut(G))]
    print report[-1]
    with open("outputs/q4/report.txt", "w") as f:
        f.write("\n".join(report))
        print "Report generated at (outputs/q4/report.txt)."
