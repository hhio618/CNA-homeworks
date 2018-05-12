import numpy as np


def load_wiki_vote():
    return np.loadtxt("data/Datasets/Wiki-Vote.txt", dtype=int)


def load_digits():
    return np.loadtxt("data/Datasets/digits", dtype=int)


def load_digits_clusters():
    return np.loadtxt("data/Datasets/realIdx", dtype=int)


def gen_wiki_vote():
    txt = np.loadtxt("data/Datasets/Wiki-Vote.txt", dtype=int)
    np.savetxt("data/Datasets/Wiki-Vote.csv", txt, fmt="%d", delimiter=",")
