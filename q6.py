import scipy
import scipy.sparse as sparse
from data import data
import matplotlib
import networkx as nx
import numpy as np
import operator
from base import pagerank

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


def global_clustering_coefficient(G):
    return round(nx.average_clustering(G), DECIMALS)


if __name__ == '__main__':
    E = data.load_wiki_vote()
    report = ""
    # calculate measures
    G = nx.DiGraph()
    G.add_edges_from(E)
    pr = nx.pagerank(G, alpha=0.9)
    pr_top10 = sorted(pr.items(), key=operator.itemgetter(1), reverse=True)[:10]
    h, a = nx.hits(G)
    hits_top10 = sorted(h.items(), key=operator.itemgetter(1), reverse=True)[:10]
    print pr
    print hits_top10
