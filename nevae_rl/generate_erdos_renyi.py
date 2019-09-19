import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt

from random import choice

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
        #G = nx.powerlaw_cluster_graph(n,m,p)
        #nx.draw_networkx(G, with_labels= True)
	print G.nodes()
        degree_sequence=list(nx.degree(G).values())
        print degree_sequence
	#print G.neighbors(0)
	#plt.axis('off')
	#plt.show()
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
            G = create_graph(params.n, params.m, params.p)
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

	

