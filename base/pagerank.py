import numpy as np
import networkx as nx
import operator


def personalization_dict(nodes, kv):
    tmp = {i: 0 for i in nodes}
    tmp[kv[0]] = 1
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
