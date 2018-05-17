import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
#from networkx.generators.tree import random_tree
from random import choice
import numpy as np

def random_walk(G, seed, k):
	walk = [seed]
	i = 0
	while (i < k):
		neighbors = G.neighbors(seed)
		# uniformly select any neighbor from the set of neighbors
		next_hop = choice(neighbors)
		walk.append(next_hop)
		seed = next_hop
		i += 1
	return walk
		
def create_graph(n,m,p):
	#G = nx.circular_ladder_graph(n)
        #G = nx.erdos_renyi_graph(n, p)
	G = nx.barabasi_albert_graph(n, m)
        #G = random_tree(n)
        #G = nx.random_powerlaw_tree(n, tries = 200)
        #G = nx.star_graph(n)
        #G = nx.balanced_tree(10,10)
        #nx.draw_networkx(G, with_labels= True)
	print G.nodes()
        degree_sequence=list(nx.degree(G).values())
        print degree_sequence
	#print G.neighbors(0)
	#plt.axis('off')
	#plt.show()
	return G
def create_graph_specified_node_edge(n, m):
    G=nx.Graph()
    for i in range(n):
        G.add_node(i)
    candidate_edges = []
    for i in range(n):
        for j in range(i+1,n):
            if i != j:
                candidate_edges.append((i,j))
   
    edges = np.random.choice(range(len(candidate_edges)), m, replace=False)
    for i in edges:
        (u,v) = candidate_edges[i]
        G.add_edge(u,v)

    return G

def get_params():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    	parser.add_argument('--n', type=int, default=10,
			help='number of nodes in the graph')
        parser.add_argument('--m', type=int, default=10,
			help='number of edges in the graph')

        parser.add_argument('--p', type=float, default=0.1,
                        help='probability')
	parser.add_argument('--k', type=int, default=5,
                        help='length of the random walk')
        parser.add_argument('--N', type=int, default=5,
                            help='Number graph with the same parameter you want to learn')
        parser.add_argument('--file', type=str, default='graph/',
                            help='File to store the graph')
	params = parser.parse_args()
	return params
	
if __name__ == "__main__":
	params = get_params()
	#fh= open("test.edgelist", "ab")
        #G = create_graph(params.n, params.p)
	#A = nx.adjacency_matrix(G)
        for i in range(params.N):
            #G = create_graph(params.n, params.m, params.p)
	    G = create_graph_specified_node_edge(params.n, params.m)
            A = nx.adjacency_matrix(G)
            fh = open(params.file+str(i)+".edgelist" , "wb")
            nx.write_edgelist(G, fh)
            fh.write("\n")
	#fh.close()
        #fh = open("test.edgelist", "rb")
        #lines = fh.read().split('\n\n')
        #for line in lines:
        #print line
        #G = nx.read_edgelist(fh)
        #A = nx.adjacency_matrix(G)
        #print A.todense()
        #print A
	
        #for n in G.nodes():	
	#	print n,random_walk(G, n, params.k)

	

