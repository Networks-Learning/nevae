import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import glob
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.layout import *
from random import choice
from collections import defaultdict
from math import log


# v is joining to u
def getprob(degseq, u, penalty):
    #print degseq
    #print u, degseq[u], sum([degseq[val] for val in degseq.keys()])

    #print degseq[u]*1.0/sum([degseq[val] for val in degseq.keys()])
    return log((degseq[u]+0.001)/(sum([degseq[val] for val in degseq.keys()])+ penalty ))

def getlikelihood1(G,n,m):
    maxnode = np.argsort(np.array(list(nx.degree(G).values())))[-1]
    #print maxnode
    edges = list(nx.bfs_edges(G, maxnode))
    #print(len(edges))
    #print edges
    logp = 0.0
    degseq = defaultdict()
    
    for u in G.nodes():
        degseq[int(u)] = 0

    degseq[edges[0][0]] = degseq[edges[0][1]] = 1

    for (u, v) in edges[1:]:
            logp += getprob(degseq, u, 0)
            degseq[int(u)] += 1
            degseq[int(v)] += 1
    #print len(G.edges())
    for i in range(len(edges)):
        (u,v) = edges[i]
        if u > v:
            edges[i] = (v, u)
    remain_edgeseq = len(G.edges()) - len(edges)
    for (u,v) in list(set(G.edges()) - set(edges)):
            logp += getprob(degseq, u, 1000)
            degseq[int(u)] += 1
            degseq[int(v)] += 1

    #print remain_edgeseq
    #logp += (remain_edgeseq * -2.0)
    return logp



def getlikelihood(G, n, m):
    degree_sequence = np.array(list(nx.degree(G).values()))
    #print degree_sequence
    sorted_seq = np.argsort(degree_sequence)
    print sorted_seq
    highestdeg = sorted_seq[-1]
    edges = G.edges(highestdeg)
    visited = [edges[0]]
    visited_node = []
    degseq = defaultdict()
    for u in G.nodes():
        degseq[int(u)] = 0

    degseq[edges[0][0]] = degseq[edges[0][1]] = 1
    logp = 0

    while (len(visited)< (len(G.edges()))):
        degree_n = []
        max_v = -1
        max_val = 0
        for (u,v) in edges:
            if u < v :
                if (u,v) in visited:
                    continue
                visited.append((u,v))
            else:
                if (v,u) in visited:
                    continue
                visited.append((v,u))
            if max_val < G.degree(v):
                max_val = G.degree(v)
                max_v = v
            logp += getprob(degseq, u)
            degseq[int(u)] += 1
            degseq[int(v)] += 1
            print len(visited), len(G.edges())
        visited_node.append(u)
        print max_v
        max_val = 0
        print set(G.edges()) - set(visited)
        '''
        if max_v == -1:
            newedges = []
            nodes = list(set(G.nodes()) - set(visited_node))
            for node in nodes:
                edges = list(set(G.edges(node)) - set(visited))
                if max_val < len(edges):
                #if max_val < G.degree(node):
                    max_val = len(edges)
                    #G.degree(node)
                    max_v = node
                    newedges = edges
                    #print max_v, max_val
            edges = newedges
            print edges, max_v, max_val
            #edges = list(set(G.edges(max_v)) - set(visited))[1:]
            degseq[edges[0][0]] += 1
            degseq[edges[0][1]] += 1
        else:
        '''
        edges = G.edges(max_v)
    return logp
    

    '''
    prob = 1
    edgelist = []
    degseq = defaultdict()
    for node in reversed(sorted_seq):
        print node, G.edges(node)
        edgelist.extend(np.array(G.edges(node)))

    for u in G.nodes():
        degseq[int(u)] = 0

    logp = 0.0
    degseq[edgelist[0][0]] += 1
    degseq[edgelist[0][1]] += 1

    for (u,v) in edgelist[1:]:
        logp += getprob(degseq, u)
        degseq[int(u)] += 1
        degseq[int(v)] += 1
        #print degseq   
    return logp 
    '''

if __name__ == "__main__":
	
        path = sys.argv[1]
        
        n = int(sys.argv[2])
        m = int(sys.argv[3])

        count = 0
        index = range(1000)
        for findex in index:
            fname = path + "/"+str(findex)+".edgelist"
            #for fname in sorted(glob.glob(path)):
            #for fname in [path]:
            #print fname
            f = open(fname, 'rb')
            G=nx.read_edgelist(f, nodetype=int)
            for i in range(n):
                if i not in G.nodes():
                    G.add_node(i)
    
            print getlikelihood1(G, n, m)
	

