import numpy as np


def load_actor_movie():
    return np.loadtxt("data/Datasets/out.actor-movie", skiprows=1, dtype=int)


def load_actor_movie_weighted():
    E = np.loadtxt("data/Datasets/out.actor-movie", skiprows=1, dtype=int)
    one = np.ones(shape=len(E))
    return np.c_[E, one]

