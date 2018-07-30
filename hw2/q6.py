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
from q5 import run
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

def clustering_seed_set(G, seed_size, p):
    print "Run KMeans clustering..."
    mm_clusters = KMeans(n_clusters=10).fit(get_data_from(graph)).labels_
    np.savetxt('outputs/q8/modularity-idx.csv', mm_clusters, delimiter=",", fmt="%d")
    print "Done"

if __name__ == '__main__':
    run("q6", clustering_seed_set)