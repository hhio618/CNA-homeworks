import numpy as np
import sys


def confusion_matrix(expected, predicted):
    classes = list(sorted(np.unique(expected)))[::-1]
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes))
    for idx in xrange(len(expected)):
        cm[classes.index(predicted[idx])][classes.index(expected[idx])] += 1
    return cm


def sens(cm):
    return float(cm[0][0]) / cm[:, 0].sum()


def spec(cm):
    return float(cm[1][1]) / cm[:, 1].sum()


def acc(cm):
    return cm.trace() / cm.sum()


class SSE:
    def __repr__(self):
        return "sse"

    def __call__(self, *args, **kwargs):
        X = args[0]
        C = args[1]
        X_copy = X.copy()
        for cluster_no in np.unique(C):
            idx = (C == cluster_no)
            X_copy[idx] = X_copy[idx] - X_copy[idx].mean(axis=0)
        return np.linalg.norm(X_copy)


class DBI:
    def __init__(self):
        self.centroids = {}

    def __repr__(self):
        return "dbi"

    def _s(self, X, C, i):
        return np.linalg.norm(X[C == i] - self.centroids[i])

    def _m(self, i, j):
        return np.linalg.norm(self.centroids[i] - self.centroids[j])

    def dbi(self, X, C):
        dbi = 0
        cluster_ids = list(np.unique(C))
        for i in cluster_ids:
            D_i = -sys.maxsize
            cluster_ids_prime = cluster_ids[:]
            cluster_ids_prime.remove(i)
            for j in cluster_ids_prime:
                R = float(self._s(X, C, i) + self._s(X, C, j)) / self._m(i, j)
                D_i = max(D_i, R)
            dbi += D_i
        return dbi / float(len(cluster_ids))

    def __call__(self, *args, **kwargs):
        X = args[0]
        C = args[1]
        for cluster_no in np.unique(C):
            idx = (C == cluster_no)
            self.centroids[cluster_no] = X[idx].mean(axis=0)
        return self.dbi(X, C)


class AverageLinkage:
    def __init__(self):
        pass

    def __repr__(self):
        return "avg"

    def __call__(self, *args, **kwargs):
        X_i = args[0]
        X_j = args[1]
        linkage = 0
        for xi in X_i:
            for xj in X_j:
                linkage += np.linalg.norm(xi - xj)
        return linkage / float(X_j.shape[0] * X_i.shape[0])


class SingleLinkage:
    def __init__(self):
        pass

    def __repr__(self):
        return "single"

    def __call__(self, *args, **kwargs):
        X_i = args[0]
        X_j = args[1]
        linkage = sys.maxsize
        for xi in X_i:
            for xj in X_j:
                linkage = min(np.linalg.norm(xi - xj), linkage)
        return linkage


class CompleteLinkage:
    def __init__(self):
        pass

    def __repr__(self):
        return "complete"

    def __call__(self, *args, **kwargs):
        X_i = args[0]
        X_j = args[1]
        linkage = -sys.maxsize
        for xi in X_i:
            for xj in X_j:
                linkage = max(np.linalg.norm(xi - xj), linkage)
        return linkage
