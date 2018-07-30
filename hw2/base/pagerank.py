import numpy as np
import networkx as nx
import operator


def personalization_dict(nodes, node):
    tmp = {i: 0 for i in nodes}
    tmp[node] = 1
    return tmp


class PageRankClustering:
    def __init__(self):
        pass

    def fit(self, graph, top_nodes_dict):
        nodes = graph.nodes()
        top_nodes_count = len(top_nodes_dict)
        scores = np.zeros((len(nodes), top_nodes_count))
        i = 0
        for kv in top_nodes_dict:
            print "Running personalized pagerank on node #%d ..." % kv[0]
            personalized_rank = nx.pagerank(graph, personalization=personalization_dict(nodes, kv))
            scores[:, i] = np.asarray(sorted(personalized_rank.items(), key=operator.itemgetter(0), reverse=True))[:, 1]
            i += 1
            print "Ok!"
        print "Find clusters based on personalized scores..."
        clusters = np.argmax(scores, axis=-1)
        print "Done"
        return clusters


class PageRankClassifier:
    def __init__(self):
        pass

    def fit(self, graph, top_nodes, labels):
        nodes = graph.nodes()
        top_nodes_count = len(top_nodes)
        scores = np.zeros(shape=(graph.number_of_nodes(), len(top_nodes)))
        i = 0
        for node in top_nodes:
            print "Running personalized pagerank on node #%d ..." % node
            personalized_rank = nx.pagerank(graph, personalization=personalization_dict(nodes, node))
            scores[:, i] = np.asarray(sorted(personalized_rank.items(), key=operator.itemgetter(0), reverse=True))[:, 1]
            i += 1
            print "Ok!"
        print "Find labels based on personalized scores..."
        x = list(np.argmax(scores, axis=-1))
        pred = [labels[int(top_nodes[xi])] for xi in x]
        print "Done"
        return np.asarray(pred)
