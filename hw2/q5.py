import os
import math
import matplotlib
import networkx as nx
import networkx.algorithms as algos
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from data import data
from base.degree_discount import degreeDiscountIC, spreadDegreeDiscount, runIC
import numpy as np
import community
import random
import itertools
import time
snap_available = True
try:
    import snap
except:
    snap_available = False
    print("Snap isn't available, fallback to networkx")

gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
for gui in gui_env:
    try:
        print("Testing matplotlib backend...", gui)
        matplotlib.use(gui, warn=False, force=True)
        matplotlib.interactive(False)
        from matplotlib import pyplot as plt

        break
    except Exception as e:
        continue


if __name__ == '__main__':
    E = data.load_actor_movie_weighted()
    # E = np.array([[1, 2], [2, 3], [3, 6], [6, 1], [7, 1]]) # for test
    report = []
    # calculate measures
    G = nx.DiGraph()
    G.add_weighted_edges_from(E)
    num_nodes = G.number_of_nodes()
    p = 0.5 # model probablity
    # x axis seed set percent
    seed_percent = np.arange(0.01, 0.11, 0.01)
    # y axis targeted set count
    target_counts = []
    # we calculate average activated set size
    iterations = 200 # number of iterations
    for percent in seed_percent:
            seed_size = int(percent*num_nodes)
            report += ["Find seed set for %d percent of graph(%d nodes)" % (int(percent*100),seed_size)]
            print(report[-1])
            
            # calculate initial set
            S = degreeDiscountIC(G, seed_size,p)
            report += ['Initial set of %d nodes chosen'%seed_size]
            print(report[-1])
            avg = 0
            for i in range(iterations):
                T = runIC(G, S, p)
                avg += float(len(T))/iterations
                # print i, 'iteration of IC'
            target_count = int(round(avg))
            report += ['Avg. Targeted %d nodes out of %d'%(target_count, len(G))]
            print(report[-1])
            target_counts.append(target_count)
    # Print out the figure
    plt.plot(seed_percent, target_counts)
    plt.title("#Initial vs #Target")
    plt.xlabel("#Initial")
    plt.ylabel("#Target")
    plt.savefig("outputs/q5/initial-target.png")
    # Find initial seed size for target of size 95 percent
    targeted_size = int(0.95*num_nodes)
    report += ["Searching for optimum seed size for target of size 95%%(%d of %d)" % (targeted_size,num_nodes)]
    print(report[-1])
    report += ["Searching using binary search..."]
    print(report[-1])
    S, Tsize = spreadDegreeDiscount(G, targeted_size, step=200, p=p)
    report += ["Find a solution #Seed: %d, #Target: %d"%(len(S), Tsize[len(S)])]
    print(report[-1])

    with open("outputs/q5/report.txt", "w") as f:
        f.write("\n".join(report))
        print("Report generated at (outputs/q5/report.txt).")

