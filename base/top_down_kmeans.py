import numpy as np
from sklearn.cluster import KMeans


class TDKMeans(object):
    def __init__(self):
        self.C = None

    # train kmeans and find clusters and centeroids
    def fit(self, func_data, count, criterion_checker):
        self.C = 0 * np.ones(count, dtype=int)
        nodes = np.arange(0, count)
        while criterion_checker.ok(self.C):
            unique_clusters = np.unique(self.C)
            for cluster_no in unique_clusters:
                idx_community = (self.C == cluster_no)
                model = KMeans(2).fit(func_data(nodes[idx_community]))
                self.C[idx_community] = model.labels_ + self.C.max() + 1  # <----- cluster number shoud be different
        return self
