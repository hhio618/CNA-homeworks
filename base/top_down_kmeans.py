import numpy as np
from sklearn.cluster import KMeans


class TDKMeans(object):
    def __init__(self):
        self.C = None

    # train kmeans and find clusters and centeroids
    def fit(self, func_data, index, criterion_checker):
        index = np.asarray(index)
        self.C = 0 * np.ones(len(index), dtype=int)
        i = 0
        while criterion_checker.ok(self.C):
            print "KMeans #clusters: %d" % 2 ** i
            print "Done"
            unique_clusters = np.unique(self.C)
            i += 1
            for cluster_no in unique_clusters:
                idx_community = (self.C == cluster_no)
                model = KMeans(2).fit(func_data(index[idx_community]))
                self.C[idx_community] = model.labels_ + self.C.max() + 1  # <----- cluster number shoud be different
        return self
