import scipy
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import os
import math
from base.top_down_kmeans import TDKMeans
from data import data
import matplotlib
import networkx as nx
import numpy as np
import community

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


def get_data_from(G):
    def _func(sub_nodes):
        print "Calculating modularity matrix and it's eigs ..."
        mm = nx.modularity_matrix(G.subgraph(sub_nodes))
        vals, vecs = scipy.linalg.eigh(mm)
        print "Done"
        count = vecs.shape[1]
        vecs = np.flip(vecs, -1)
        return vecs[:, :min(math.ceil(count / 2.0), 30)]

    return _func


class ModularityCriterion:
    def __init__(self, G):
        self.G = G
        self.best_modularity = -1

    def ok(self, labels):
        parts = dict(zip(self.G.nodes(), labels))
        modularity = community.modularity(parts, self.G)

        if self.best_modularity <= modularity:
            self.best_modularity = modularity
            print "New best modularity: %f" % modularity
            return True
        return False


if __name__ == '__main__':
    nodes = data.load_digits()
    true_clusters = data.load_digits_clusters()
    report = []
    num_nodes = nodes.shape[0]
    tm = []

    print "Create edges from cosine similarity..."
    if not os.path.exists("outputs/q7/intermediates/cosine.npy"):
        for idx1 in range(num_nodes):
            for idx2 in range(idx1 + 1, num_nodes):
                tm.append([idx1, idx2, cosine(nodes[idx1], nodes[idx2])])
        tm = np.asarray(tm)
        np.save('outputs/q7/intermediates/cosine.npy', tm)
    else:
        tm = np.load('outputs/q7/intermediates/cosine.npy')
    print "Done"

    print "Finding best lambda..."
    lmbda = tm[:, 2].mean()
    report += ["Lambda: %f" % lmbda]
    print report[-1]
    print "Done"

    tm = tm[tm[:, 2] >= lmbda]
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(0, nodes.shape[0]))
    graph.add_edges_from(tm[:, :2].astype('int').tolist())

    print "Run KMeans clustering..."
    model = TDKMeans()
    mm_clusters = model.fit(get_data_from(graph), nodes.shape[0], ModularityCriterion(graph))
    np.savetxt('outputs/q7/modularity-idx.csv', mm_clusters.C, delimiter=",", fmt="%d")
    report += ["TopDownKMeans --> outputs/q7/modularity-idx.csv"]
    nmi = normalized_mutual_info_score(mm_clusters.C, true_clusters)
    report += ["NMI(modularity matrix): %f" % nmi]
    print report[-1]
    print "Done"

    print "Run pagerank clustering..."
    # TODO pagerank clustering
    print "Done"

    with open("outputs/q7/report.txt", "w") as f:
        f.write("\n".join(report))
        print "Report generated at (outputs/q7/report.txt)."

    print "Saving output graph to outputs/q7/graph.png..."
    options = {
        'node_color': 'blue',
        'node_size': 1,
        'line_color': 'red',
        'linewidths': 0,
        'width': 0.02,
    }
    nx.draw_spring(graph, **options)
    plt.savefig("outputs/q7/graph.png")
    print "Done"
