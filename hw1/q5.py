import scipy
from scipy.spatial.distance import cosine
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
        count = vecs.shape[1]
        vecs = np.flip(vecs, -1)
        ub = max(min(math.ceil(count / 2.0), 30), 1)
        return vecs[:, :int(ub)]  # our method!

    return _func


class ModularityCriterion:
    def __init__(self, G):
        self.G = G
        self.best_modularity = -1
        self.modularity_list = []
        self.best_labels = None

    def ok(self, labels):
        parts = dict(zip(self.G.nodes(), labels))
        modularity = community.modularity(parts, self.G)
        if self.best_modularity <= modularity:
            self.best_modularity = modularity
            print "New best modularity: %f" % modularity
            self.best_labels = labels.copy()
            self.modularity_list.append(modularity)
            return True
        else:
            print "Modularity decreasing, rollback..."
            print "Remapping indexes..."
            uniques = np.unique(self.best_labels)
            sorted_idx = range(0, len(uniques))
            labels_new = self.best_labels.tolist()
            remap = dict(zip(uniques, sorted_idx))
            labels_new = map(lambda x: remap[x], labels_new)
            self.best_labels = np.asarray(labels_new)
            return False


if __name__ == '__main__':
    edges = data.load_wiki_vote()
    report = []
    num_nodes = edges.shape[0]
    graph = nx.Graph()
    graph.add_edges_from(edges)

    print "Run KMeans clustering..."
    model = TDKMeans()
    criterion = ModularityCriterion(graph)
    model.fit(get_data_from(graph), graph.nodes(), criterion)
    np.savetxt('outputs/q5/modularity-idx.csv', criterion.best_labels, delimiter=",", fmt="%d")
    report += ["TopDownKMeans --> outputs/q5/modularity-idx.csv"]
    print "Done"

    print "Saving output graph to outputs/q5/graph.png..."
    num_points = len(criterion.modularity_list)
    xx = [2 ** i for i in range(num_points)]
    points = xx, criterion.modularity_list
    report += ["Modularity vs clusters: \n" + str(zip(*points))]
    plt.plot(points[0], points[1])
    plt.title("Modularity vs clusters")
    plt.xlabel("Clusters")
    plt.ylabel("Modularity")
    plt.savefig("outputs/q5/graph.png")
    print "Done"

    with open("outputs/q5/report.txt", "w") as f:
        f.write("\n".join(report))
        print "Report generated at (outputs/q5/report.txt)."
