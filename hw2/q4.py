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
        params = np.loadtxt("outputs/q4/tmp/%s.txt" % G.__hash__())
    except Exception as e:
        num_nodes = len(G.nodes())
        ccoef = float(clustering_coef(G))
        in_avg = reduce(lambda a, b: a+b,
                        map(lambda x: x[1], G.in_degree()))/float(num_nodes)
        out_avg = reduce(lambda a, b: a+b,
                        map(lambda x: x[1], G.out_degree()))/float(num_nodes)
        max_d = -sys.maxsize
        G_undirected = G.to_undirected()
        for c in nx.connected_components(G_undirected):
            d = G_undirected.subgraph(c).diameter()
            if d> max_d:
                max_d = d
        params = [ccoef, in_avg, out_avg, max_d]
        np.savetxt(params, "outputs/q4/tmp/%s.txt" % G.__hash__())
    return params


class DFSSampler:
    def __init__(self, G):
        self.G = G

    def sample(self, percent):
        num_sample_edges = int(percent * self.G.number_of_edges())
        start_node = random.sample(self.G.nodes(), 1)[0]
        sampled_edges = self.dfs(start_node, num_sample_edges)
        G = nx.DiGraph()
        G.add_edges_from(sampled_edges)
        return G

    def dfs(self, start_node, num_sample_edges):
        sampled_edges = []
        for edge in nx.edge_dfs(self.G, start_node):
            if len(sampled_edges) < num_sample_edges:
                sampled_edges.append(edge)
            else:
                break
        return sampled_edges

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
        start_node = random.sample(self.G.nodes(), 1)[0]
        v = start_node
        nodes = [start_node]
        for c in itertools.count():
            if c == size:
                break
            p = random.random()
            neighbors = list(self.G.neighbors(v))
            # Get stucking
            if len(neighbors) == 0:
                random.sample(self.G.nodes(), 1)[0]
                continue
            candidate = random.sample(self.G.neighbors(v), 1)[0]
            if candidate in nodes:
                random.sample(self.G.nodes(), 1)[0]
                continue
            v = candidate
            nodes.append(v)
        return nodes

    def __str__(self):
        return "RWSampler"


class FFSampler:
    def __init__(self, G):
        self.G = G

    def sample(self, percent):
        num_sample_nodes = int(percent * self.G.number_of_nodes())
        sampled_nodes = self.forest_fire(num_sample_nodes)
        return self.G.subgraph(sampled_nodes)

    def forest_fire(self, pf, size=-1):
        p = float(1-pf)/pf
        start_node = random.sample(self.G.nodes(), 1)[0]
        v = start_node
        nodes = [start_node]

        def _forest_fire():
            x = np.random.geometric(p=p)
            neighbors = list(self.G.neighbors(v))
            unvisited_neighbors = [x for x in neighbors if not(x in nodes)]
            nodes += unvisited_neighbors
            for w in unvisited_neighbors:
                _forest_fire(w)
        return nodes

    def __str__(self):
        return "NodeSampler"


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
    samplers = [NodeSampler(G),\
            EdgeSampler(G), DFSSampler(G)]
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
    indics = ["*b","^r","oy","xg","-p"]
    for param,indic in zip(param_names,indics):
        plt.figure()
        # Draw original parameters
        yy = [org_params[param_names.index(param)]]*len(xx)
        plt.plot(xx, yy,indic,s=10,label="original")

        for sampler in samplers:
            xx = percentages
            yy = parameters[str(sampler)][param]
            # Print out the figure
            plt.plot(xx, yy,indic, label=str(sampler))
            plt.title("%s vs sampling percentage"% param)
            plt.xlabel("Sampling percentage")
            plt.ylabel(param)
            plt.legend()
            plt.savefig("outputs/q4/%s.png"%param)
    with open("outputs/q4/report.txt", "w") as f:
        f.write("\n".join(report))
        print "Report generated at (outputs/q4/report.txt)."