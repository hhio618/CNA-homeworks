import os, sys
import math
import matplotlib
import networkx as nx
import networkx.algorithms as algos
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from data import data
import numpy as np
import community
import random
import itertools
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


# Create parameters dictionary
def create_param_dict(samplers, *args):
    d = dict()
    for sampler in samplers:
        d[str(sampler)] = dict()
        for arg in args:
            d[str(sampler)][arg] = list()
    return d


def clustering_coef(G):
        # Betweenness coefficient
    # if snap_available:
    #     G_snap = snap.LoadEdgeList(snap.TNGraph, "data/Datasets/Wiki-Vote.txt", 0, 1)
    #     average_clustering_coef = snap.GetClustCf(G_snap, -1)
    #     report += ["Average clustering coefficient: %f" % float(average_clustering_coef)]
    # else:
        # Clustering coefficient
        return nx.average_clustering(G.to_undirected())


def calculate_params(G):
    params = []
    try:
        params = np.loadtxt("outputs/q4/tmp/h%s.txt" % G.__hash__())
    except Exception as e:
        num_nodes = len(G.nodes())
        ccoef = float(clustering_coef(G))
        in_avg = reduce(lambda a, b: a+b,
                        map(lambda x: x[1], G.in_degree()))/float(num_nodes)
        out_avg = reduce(lambda a, b: a+b,
                        map(lambda x: x[1], G.out_degree()))/float(num_nodes)
        Ghat = G.to_undirected()
        d = 0
        # i = 0
        # for c in nx.connected_components(Ghat):
        #     Ghat_comp = Ghat.subgraph(c)
        #     d += nx.average_shortest_path_length(Ghat_comp)
        #     i+=1
        # d = d/float(i)
        
        params = [ccoef, in_avg, out_avg, d]
        np.savetxt("outputs/q4/tmp/h%s.txt" % G.__hash__(), params)
    return params


class DFSSampler:
    def __init__(self, G):
        self.G = G

    def sample(self, percent):
        num_sample_nodes = int(percent * self.G.number_of_nodes())
        start_node = random.sample(self.G.nodes(), 1)[0]
        sampled_nodes = self.dfs(start_node, num_sample_nodes)
        return self.G.subgraph(sampled_nodes)

    def dfs(self, start_node, num_sample_nodes):
        return nx.dfs_preorder_nodes(self.G,start_node, depth_limit=num_sample_nodes)

    def __str__(self):
        return "DFSSampler"


class RWSampler:
    def __init__(self, G):
        self.G = G

    def sample(self, percent):
        num_sample_nodes = int(percent * self.G.number_of_nodes())
        sampled_nodes = list(self.random_walk(size=num_sample_nodes))
        return self.G.subgraph(sampled_nodes)

    def random_walk(self, size=-1):
        pr = nx.pagerank(self.G)
        return pr.keys()[:size]

    def __str__(self):
        return "RWSampler"


class FFSampler:
    def __init__(self, G):
        self.G = G

    def sample(self, percent):
        num_sample_nodes = int(percent * self.G.number_of_nodes())
        sampled_nodes = self.forest_fire(pf=0.5, size=num_sample_nodes)
        return self.G.subgraph(sampled_nodes)

    def forest_fire(self, pf, size=-1):
        p = float(1-pf)/pf
        start_node = random.sample(self.G.nodes(), 1)[0]
        nodes = [start_node]

        def _forest_fire(v, nodes):
            if len(nodes) > size:
                return
            x = np.random.geometric(p=p)
            neighbors = list(self.G.neighbors(v))
            unvisited_neighbors = [item for item in neighbors if not(x in nodes)]
            nodes += unvisited_neighbors
            for w in unvisited_neighbors:
                _forest_fire(w, nodes)
        _forest_fire(start_node, nodes)
        return nodes

    def __str__(self):
        return "FFSampler"


class NodeSampler:
    def __init__(self, G):
        self.G = G

    def sample(self, percent):
        num_sample_nodes = int(percent * self.G.number_of_nodes())
        sampled_nodes = random.sample(self.G.nodes(), num_sample_nodes)
        return self.G.subgraph(sampled_nodes)

    def __str__(self):
        return "NodeSampler"


class EdgeSampler:
    def __init__(self, G):
        self.G = G

    def sample(self, percent):
        num_sample_edges = int(percent * self.G.number_of_edges())
        sampled_edges = random.sample(self.G.edges(), num_sample_edges)
        G = nx.DiGraph()
        G.add_nodes_from(G.nodes())
        G.add_edges_from(sampled_edges)
        return G

    def __str__(self):
        return "EdgeSampler"


if __name__ == '__main__':
    E = data.load_actor_movie()
    # E = np.array([[1, 2], [2, 3], [3, 6], [6, 1], [7, 1]]) # for test
    report = []
    # calculate measures
    G = nx.DiGraph()
    G.add_edges_from(E)
    org_params = [0.1,2.,2.,5.]#calculate_params(G)
    samplers = [NodeSampler(G),RWSampler(G),
            EdgeSampler(G), DFSSampler(G),FFSampler(G)]
    # For saving hyper parameters
    param_names = ['ccoef', 'in', 'out', 'd']
    parameters = create_param_dict(samplers, *param_names)
    # x axis (percentage)
    percentages = [.05, .1, .15, .2]
    for percent in percentages:
        report += ["Sampling for %d percent of graph" % int(percent*100)]
        print(report[-1])
        for sampler in samplers:
            report += ["Using sampler %s" % sampler]
            print(report[-1])
            G = sampler.sample(percent)
            ccoef, in_avg, out_avg, diameter = calculate_params(G)
            parameters[str(sampler)]['ccoef'].append(ccoef)
            parameters[str(sampler)]['in'].append(in_avg)
            parameters[str(sampler)]['out'].append(out_avg)
            parameters[str(sampler)]['d'].append(diameter)
            report += ["Average clustering coefficient: %f" % ccoef]
            print report[-1]

    # Draw Sampled graph parameters vs Original graph
    indics = ["*b-","^r:","oy--","xg-.","pk-"]
    xx = percentages
    for param in param_names:
        plt.figure()
        # Draw original parameters
        yy = [org_params[param_names.index(param)]]*len(xx)
        plt.plot(xx, yy, "k-",linewidth=2, label="original")
        for sampler, indic in zip(samplers, indics):
            yy = parameters[str(sampler)][param]
            # Print out the figure
            plt.plot(xx, yy,indic,linewidth=2, label=str(sampler))
            plt.title("%s vs sampling percentage"% param)
            plt.xlabel("Sampling percentage")
            plt.ylabel(param)
            plt.legend()
            plt.savefig("outputs/q4/%s.png"%param)
    with open("outputs/q4/report.txt", "w") as f:
        f.write("\n".join(report))
        print "Report generated at (outputs/q4/report.txt)."
