from sklearn.metrics import normalized_mutual_info_score
from base.pagerank import PageRankClustering
from base.utils import calculate_modularity
from data import data
import matplotlib
import networkx as nx
import numpy as np
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

if __name__ == '__main__':
    E = data.load_wiki_vote()
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    report = []
    # calculate measures
    G = nx.DiGraph()
    G.add_edges_from(E)
    print "Calculating pagerank ..."
    pr = nx.pagerank(G, alpha=0.9)
    pr_nodes_sorted = np.asarray(sorted(pr.items(), key=operator.itemgetter(1), reverse=True))
    print "Done"
    print "Calculating HITS ..."
    h, a = nx.hits(G)
    hits_nodes_sorted = np.asarray(sorted(h.items(), key=operator.itemgetter(1), reverse=True))
    print "Done"
    plt.plot(pr_nodes_sorted[:100, 1], label="PageRank")
    plt.plot(hits_nodes_sorted[:100, 1], label="HITS")
    plt.legend(loc='upper right')
    plt.xlabel("Top nodes")
    plt.ylabel("Scores")
    plt.savefig("outputs/q6/scores.png")  # 15 based on figures
    k = 15
    report += ["PageRank top nodes --> " + np.array2string(pr_nodes_sorted[:k], separator=",")]
    report += ["HITS top nodes --> " + np.array2string(pr_nodes_sorted[:k], separator=",")]

    print "Clustering using proposed model (pagerank)..."
    model = PageRankClustering()
    clusters_pr = model.fit(graph=G, top_nodes_dict=pr_nodes_sorted[:k])
    pr_modularity = calculate_modularity(graph=G, clusters=clusters_pr)
    report += ["PageRank modularity: %f" % pr_modularity]
    print report[-1]
    np.savetxt("outputs/q6/pr-idx.csv", clusters_pr, delimiter=",", fmt="%d")
    report += ["PageRank clustering result --> outputs/q6/pr-idx.csv"]
    print report[-1]
    print "Done"
    print "Clustering using proposed model (HITS)..."
    clusters_hits = model.fit(graph=G, top_nodes_dict=hits_nodes_sorted[:k])
    hits_modularity = calculate_modularity(graph=G, clusters=clusters_hits)
    report += ["HITS modularity: %f" % hits_modularity]
    print report[-1]
    np.savetxt("outputs/q6/hits-idx.csv", clusters_hits, delimiter=",", fmt="%d")
    report += ["HITS clustering result --> outputs/q6/pr-hits.csv"]
    print report[-1]
    print "Done"
    nmi = normalized_mutual_info_score(clusters_pr, clusters_hits)
    report += ["NMI (HITS vs PageRank): %f" % nmi]
    print report[-1]

    with open("outputs/q6/report.txt", "w") as f:
        f.write("\n".join(report))
        print "Report generated at (outputs/q6/report.txt)."
