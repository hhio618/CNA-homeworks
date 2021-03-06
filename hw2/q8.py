import scipy
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import os
from base.pagerank import PageRankClassifier

from data import data
import matplotlib
import networkx as nx
import numpy as np
import community
import operator

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

def count_iterable(i):
    return sum(1 for e in i)

def get_data_from(G):
    print "Calculating modularity matrix and it's eigs ..."
    mm = nx.modularity_matrix(G)
    vals, vecs = scipy.linalg.eigh(mm)
    print "Done"
    vecs = np.flip(vecs, -1)
    ub = 10
    return vecs[:, :int(ub)]  # our method!


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

def select_node(pr_sorted,labels, k):
    label_set = []
    node_list = []
    for node,rank in pr_sorted:
        if len(node_list) == k:
           return np.asarray(node_list)
        label = labels[node] 
        if not label in label_set:
            node_list.append([node, rank])
            label_set.append(label)

    

def calculate_lambda_edges(nodes, tm):
    X = []
    lmbda = None
    first_change = True
    for l in np.arange(0.0, 1, 0.05):
        tm_copy = tm[tm[:, 2] >= l]
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(0, nodes.shape[0]))
        graph.add_edges_from(tm_copy[:, :2].astype('int').tolist())
        num_isolate_node = count_iterable(nx.isolates(graph))
        if num_isolate_node > 0 and first_change:
            lmbda = l
            first_change = False
        X.append([l, num_isolate_node])
    return np.asarray(X), lmbda


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    nodes = data.load_digits()
    true_clusters = data.load_digits_clusters()
    report = []
    num_nodes = nodes.shape[0]
    tm = []

    print("Create edges from cosine similarity...")
    if not os.path.exists("outputs/q8/intermediates/cosine.npy"):
        for idx1 in range(num_nodes):
            for idx2 in range(idx1 + 1, num_nodes):
                tm.append([idx1, idx2, cosine(nodes[idx1], nodes[idx2])])
        tm = np.asarray(tm)
        np.save('outputs/q8/intermediates/cosine.npy', tm)
    else:
        tm = np.load('outputs/q8/intermediates/cosine.npy')
    print("Done")

    print("Finding best lambda...")
    X, lmbda = calculate_lambda_edges(nodes, tm)
    report += ["Saving output lambda finder graph to outputs/q8/lambda-finder.png..."]
    print(report[-1])
    report += ["Lambda vs #isolate nodes --> " + np.array2string(X, separator=",")]
    plt.plot(X[:, 0], X[:, 1])
    plt.title("Lambda finder")
    plt.xlabel("Lambda")
    plt.ylabel("#Isolate nodes")
    plt.savefig("outputs/q8/lambda-finder.png")
    print("Done")
    # lmbda = 1.4 * tm[:, 2].mean()
    tm = tm[tm[:, 2] >= lmbda]
    report += ["Best Lambda: %f" % lmbda]
    print(report[-1])
    print("Done")

    graph = nx.Graph()
    graph.add_nodes_from(np.arange(0, nodes.shape[0]))
    graph.add_edges_from(tm[:, :2].astype('int').tolist())
    k = 10

    print("Clustering using proposed model (pagerank)...")
    print("Calculating pagerank ...")
    pr = nx.pagerank(graph)
    pr_nodes_sorted = select_node(sorted(pr.items(), key=operator.itemgetter(1), reverse=True),true_clusters,k)
    print("Done")
    report += ["PageRank top nodes --> s" + np.array2string(pr_nodes_sorted, separator=",")]
    model = PageRankClassifier()
    clusters_pr = model.fit(graph=graph, top_nodes=pr_nodes_sorted[:,0],labels=true_clusters)
    acc = accuracy_score(clusters_pr, true_clusters)
    report += ["Accuracy(PageRank clustering): %f" % acc]
    print(report[-1])
    print("Done")

    with open("outputs/q8/report.txt", "w") as f:
        f.write("\n".join(report))
        print("Report generated at (outputs/q8/report.txt).")
