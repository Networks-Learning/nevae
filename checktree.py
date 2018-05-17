import os
import pickle
from numpy import *
import numpy as np
import networkx as nx
import ast
from numpy.linalg import svd, qr, norm
import glob
import sys
from collections import defaultdict

def load_data(filename, num=0):
    path = filename
    adjlist = []
    featurelist = []
    edgelist = []
    bfsedges = []
    cyclelist = []

    for fname in sorted(glob.glob(path+"*")):
        f = open(fname, 'r')
        print fname
        G=nx.read_edgelist(f, nodetype=int)
        f.close()
        n = num

        for i in range(n):
            if i not in G.nodes():
                G.add_node(i)
        degreemat = np.zeros((n,1), dtype=np.float)
        edges = G.edges()
        #print edges
        bfs_edges = list(nx.bfs_edges(G,0))
        
        try:
            adjlist.append(np.array(nx.adjacency_matrix(G).todense()))
            featurelist.append(degreemat)
            bfsedges.append((len(edges) - len(bfs_edges), edges))
            cyclelist.append(len(nx.cycle_basis(G)))
        except:
            continue
    return (adjlist, bfsedges, cyclelist)
    #return (nx.adjacency_matrix(G).todense(), degreemat, edges, non_edges)

num = int(sys.argv[2])
adjlist, bfsedges, cyclelist = load_data(sys.argv[1], num)

count = 0
disconnect = 0
cycle = 0
less = 0 
for c in cyclelist:
    #print c
    if c > 0:
        cycle+=1
for (l, e) in bfsedges:
    #print l, e
    if l > 0:
        disconnect +=1
total = 0
triangle = defaultdict(int)
#triange_count = 0
for adj in adjlist:
    triangle_count = 0
    flag = False
    #print(flag)
    #print adj, bfsedges[total][1]
    for i in range(num):
        for j in range(i+1, num):
            
            if adj[i][j] == 1:
                for k in range(num):
                    if adj[j][k] == 1 and adj[i][k] == 1:
                        print total,i, j,k
                        flag = True
                        triangle_count += 1
                        #break
            #if flag:
            #    break
    if flag:
        count += 1
        #break
    total += 1
    triangle[triangle_count]+=1
    print "Triange count", triangle_count
    #break
print "With triangle", count, "total", total, len(bfsedges)
print "Disconnected", disconnect, "Less", less
print "Cycle", cycle
for key in triangle.keys():
    print key, triangle[key]
