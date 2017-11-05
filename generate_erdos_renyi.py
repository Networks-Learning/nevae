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
		
def create_graph(n,p):
	G = nx.erdos_renyi_graph(n, p)
	#nx.draw_networkx(G, with_labels= True)
	print G.nodes()
	print G.neighbors(0)
	#plt.axis('off')
	#plt.show()
	return G

def get_params():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    	parser.add_argument('--n', type=int, default=10,
			help='number of nodes in the graph')
        parser.add_argument('--p', type=float, default=0.1,
                        help='probability')
	parser.add_argument('--k', type=int, default=5,
                        help='length of the random walk')

	params = parser.parse_args()
	return params
	
if __name__ == "__main__":
	params = get_params()
	G = create_graph(params.n, params.p)
	A = nx.adjacency_matrix(G)
	nx.write_edgelist(G, "test.edgelist")
	#print A
	
        #for n in G.nodes():	
	#	print n,random_walk(G, n, params.k)

	

