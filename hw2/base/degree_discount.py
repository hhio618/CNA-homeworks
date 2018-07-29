''' Implementation of single discount heuristic[1] for Independent Cascade model
of influence propagation in graph G
[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
'''
from pq import PriorityQueue as PQ # priority queue
import math

import networkx as nx
def runIC (G, S, p = .01):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    from random import random
    T = deepcopy(S) # copy already selected nodes

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                    # print(T[i], 'influences', v)
                    T.append(v)
        i += 1
    return T
    
def singleDiscount(G, k, p=.1):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    S = [] # set of activated nodes
    d = PQ() # degrees
    for u in G:
        degree = sum([G[u][v]['weight'] for v in G[u]])
        d.add_task(u, -degree)
    for i in range(k):
        u, priority = d.pop_item()
        S.append(u)
        for v in G[u]:
            if v not in S:
                [priority, count, task] = d.entry_finder[v]
                d.add_task(v, priority + G[u][v]['weight']) # discount degree by the weight of the edge
    return S

''' Implementation of degree discount heuristic [1] for Independent Cascade model
of influence propagation in graph G
[1] -- Wei Chen et al. Efficient influence maximization in Social Networks (algorithm 4)
'''

def degreeDiscountIC(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    for i in range(k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p # discount of degree
                dd.add_task(v, -priority)
    return S

def degreeDiscountIC2(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (without priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    Note: the routine runs twice slower than using PQ. Implemented to verify results
    '''
    d = dict()
    dd = dict() # degree discount
    t = dict() # number of selected neighbors
    S = [] # selected set of nodes
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        u, ddv = max(dd.iteritems(), key=lambda k,v: v)
        dd.pop(u)
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight'] # increase number of selected neighbors
                dd[v] = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p
    return S


def binarySearchBoundary(G, k, Tsize, targeted_size, step, p, iterations):
    # initialization for binary search

    R = iterations
    stepk = -int(math.ceil(float(step)/2))
    k += stepk
    if k not in Tsize:
        S = degreeDiscountIC(G, k, p)
        avg = 0
        for i in range(R):
            T = runIC(G, S, p)
            avg += float(len(T))/R
        Tsize[k] = avg
    # check values of Tsize in between last 2 calculated steps
    while stepk != 1:
        print(k, stepk, Tsize[k])
        if Tsize[k] >= targeted_size:
            stepk = -int(math.ceil(float(abs(stepk))/2))
        else:
            stepk = int(math.ceil(float(abs(stepk))/2))
        k += stepk

        if k not in Tsize:
            S = degreeDiscountIC(G, k, p)
            avg = 0
            for i in range(R):
                T = runIC(G, S, p)
                avg += float(len(T))/R
            Tsize[k] = avg
    return S, Tsize

def spreadDegreeDiscount(G, targeted_size, step=1, p=.01, iterations=200):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    targeted_size -- desired size of targeted set
    step -- step after each to calculate spread
    p -- propagation probability
    R -- number of iterations to average influence spread
    Output:
    S -- seed set that achieves targeted_size
    Tsize -- averaged targeted size for different sizes of seed set
    '''

    Tsize = dict()
    k = 0
    Tsize[k] = 0
    R = iterations

    while Tsize[k] <= targeted_size:
        k += step
        S = degreeDiscountIC(G, k, p)
        avg = 0
        for i in range(R):
            T = runIC(G, S, p)
            avg += float(len(T))/R
        Tsize[k] = avg

        print(k, Tsize[k])

    # binary search for optimal solution
    return binarySearchBoundary(G, k, Tsize, targeted_size, step, p, iterations)

