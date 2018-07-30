import os
import sys
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
import copy

from sklearn.metrics import precision_score
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


class Node:
    def __init__(self, id, parent, sim_value, link_value=(1.0,1.0), childs=None, leaf=False):
        #Only leaf has id's
        self.id = id
        self.parent = parent
        self.sim_value = sim_value
        self.link_value = link_value
        self.childs = childs or []
        self.leaf = leaf
    
    def __repr__(self):
        return ("Node<*" if self.leaf else "Node<") + "id:%d,l:%s>" %(self.id, self.link_value)
    def __str__(self, level=0):
        ret = "\t"*level+ ("*" if self.leaf else "") +repr(self.link_value)+"\n"
        for child in self.childs:
            ret += child.__str__(level+1)
        return ret

    def all_parents(self):
        u = self.parent
        parents = []
        while u != None:
             parents.append(u)
             u = u.parent
        return parents
    def lca(self, u):
         this_all_parent = self.all_parents()
         u_all_parent = u.all_parents()
         
         for x in this_all_parent: 
            if x in u_all_parent: #for any item 'x' from collection 'i', find the same item in collection of 'j'
                return x # print out the results

         
    def leafs(self):
        leafs = []
        def _get_leaf_nodes(node):
                if node.leaf:
                    leafs.append(node)
                else:
                    for n in node.childs:
                        _get_leaf_nodes(n)
        _get_leaf_nodes(self)
        return leafs


class Dendogram:
    def __init__(self, G, S):
        self.S = S
        self.G = G
        self.root = Node(-1,None, -1)
        # Leafs are in the id:node format
        self.leafs = {}

    def sim(self, g1, g2):
        g1_nodes = g1.leafs()
        g2_nodes = g2.leafs()
        count = len(g1_nodes) * len(g2_nodes)
        out = 0.0
        for u in g1_nodes:
            for v in g2_nodes:
               out += S[u.id, v.id]
        return out/count
    
    def _node(self, u):
        return self.leafs[u]
    # u ,v are leafs
    def link_prob(self, tup):
        u, v = tup
        n1 = self._node(u).lca(self._node(v))
        return n1.link_value[0]/n1.link_value[1]

    def link_strengh(self, g1, g2):
        g1_nodes = g1.leafs()
        g2_nodes = g2.leafs()
        possible_links = float(len(g1_nodes) * len(g2_nodes))
        real_links = 0.0
        for u in g1_nodes:
            for v in g2_nodes:
               real_links += 1.0 if self.G.has_edge(u.id, v.id) or self.G.has_edge(v.id, u.id) else 0.0
        return (real_links, possible_links)

    def likelihood(self):
        def _likelihood(node):
            if node.leaf:
                return 0.0
            l = 0.0
            for child in node.childs:
                l += np.log(_likelihood(child))
            (real_links, possible_links) = node.link_value
            return l + np.log(real_links/possible_links) * real_links +  np.log(1.0-real_links/possible_links) * (possible_links-real_links)
        return np.log_likelihood(self.root)

    def __str__(self):
        return str(self.root)

    def create(self):
        num_nodes = self.S.shape[0]
        for i in xrange(num_nodes):
            self.root.childs.append(
                Node(i, self.root, -1, leaf=True))
        
        num_groups = len(self.root.childs)
        while num_groups != 1:
            best_similarity = -sys.maxsize
            best_pair = None
            for i in xrange(num_groups):
                for j in xrange(i, num_groups):
                    if i != j:
                        sim = self.sim(self.root.childs[i], self.root.childs[j])
                        if best_similarity < sim:
                            best_similarity = sim
                            best_pair = (i, j)
            i, j = best_pair
            node_i = self.root.childs[i]
            node_j = self.root.childs[j]
            # Delete items from root
            min_index = min(i,j)
            max_index = max(i,j)
            del self.root.childs[max_index]
            del self.root.childs[min_index]
            inner_node = Node(-1, self.root, best_similarity,
                              self.link_strengh(node_i, node_j), [node_i, node_j])
            node_i.parent = inner_node
            node_j.parent = inner_node
            self.root.childs.append(inner_node)
            num_groups = len(self.root.childs)
        
        self.root = self.root.childs[0]
        # Find leafs
        for leaf in self.root.leafs():
            self.leafs[leaf.id] = leaf
        return self


class KATZSimilarity:
    def __init__(self, G, beta):
        self.G = G
        self.beta = beta

    def run(self):
        num_nodes = len(self.G.nodes())
        I = np.eye(num_nodes)
        A = nx.adjacency_matrix(self.G)
        return np.linalg.inv(I-self.beta*A) - I

    def __str__(self):
        return "KATZSimilarity"

class JCSimilarity:
    def __init__(self, G):
        self.G = G

    def run(self):
        num_nodes = len(self.G.nodes())
        S = np.zeros(shape=(num_nodes, num_nodes))
        for i in xrange(num_nodes):
            for j in xrange(i, num_nodes):
                if i == j:
                    S[i, j] = 0
                else:
                    di = G.degree[i]
                    dj = G.degree[j]
                    cn = len(list(nx.common_neighbors(self.G,i,j)))
                    S[i, j] = S[j, i] = cn/float(di + dj-cn)
        return S

    def __str__(self):
        return "JCSimilarity"

class ShortestPathSimilarity:
    def __init__(self, G):
        self.G = G

    def run(self):
        num_nodes = len(self.G.nodes())
        S = np.zeros(shape=(num_nodes, num_nodes))
        for i in xrange(num_nodes):
            for j in xrange(i, num_nodes):
                if i == j:
                    S[i, j] = 0
                else:
                    S[i, j] = S[j, i] = (len(nx.shortest_path(
                        self.G, i, j))+len(nx.shortest_path(self.G, i, j)))/2.0
        return S

    def __str__(self):
        return "ShortestPathSimilarity"

def train_test_split(G, percent):
        num_sample_edges = int(percent * G.number_of_edges())
        shuffled_edges = list(G.edges())
        random.shuffle(shuffled_edges)
        G_train = nx.Graph()
        G_train.add_nodes_from(G.nodes())
        G_train.add_edges_from(shuffled_edges[:num_sample_edges])
        E_test_pos = shuffled_edges[num_sample_edges:]
        # Negative sampling for test
        shuffled_non_edges = list(nx.complement(G).edges())
        random.shuffle(shuffled_non_edges)
        E_test_neg = shuffled_non_edges[:len(E_test_pos)]
        y_test = [1]*len(E_test_pos) + [0]*len(E_test_neg)
        E_test = E_test_pos+E_test_neg
        return G_train, E_test, np.asarray(y_test)

def balance(G):
    edges = G.edges()
    non_edges = nx.complement(G).edges()
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    max_edge = num_nodes*(num_nodes-1)/2.0
    # Balance number of edges
    new_edge = random.sample(non_edges,int(max_edge/2.0-num_edges))
    G.add_edges_from(new_edge)
    return G

def precision(true_labels, preds):
    # check if greater than 0.5 then + else -
    preds = (preds>=0.5)*1
    return precision_score(true_labels, preds,labels=[0,1])


def link_prediction(D,edge_list):
    predictions = []
    for edge in edge_list:
        predictions.append(D.link_prob(edge))
    return predictions


random.seed(0)
if __name__ == '__main__':
    E = random.sample(data.load_actor_movie(),1000)
    G = nx.Graph()
    G.add_edges_from(E)
    # # balance the imbalanced data 65 , 35 percent
    G = balance(G)
    report = []
    G_train, E_test, y_test = train_test_split(G,0.7)
    similaritie = [ShortestPathSimilarity(G_train),JCSimilarity(G_train), KATZSimilarity(G_train, 0.8)]
    liklihoods = []
    predictions = []
    for similarity in similaritie:
            report += ["Using similarity %s" % similarity]
            print(report[-1])
            S = similarity.run()
            D = Dendogram(G_train,S).create()
            print(D)
            predictions += [link_prediction(D, E_test)]
            liklihoods +=[D.likelihood()]
            report += ["Dendogram liklihood: "+ str(liklihoods[-1])]
            print(report[-1])
    
    # weighted sum over all models 
    liklihoods = np.asarray(liklihoods)
    predictions = np.asarray(predictions)
    weighted_predictions = (predictions.T*liklihoods).T.sum(axis=0)/liklihoods.sum()
    print(predictions,weighted_predictions)
    report += ["Presicion: "+ str(precision(y_test, weighted_predictions))]
    print(report[-1])