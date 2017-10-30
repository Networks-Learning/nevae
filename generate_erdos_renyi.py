import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt

def create(n,p):
	G = nx.erdos_renyi_graph(n, p)
	nx.draw_networkx(G, with_labels= True)
	print G.nodes()
	plt.axis('off')
	plt.show()

def getparams():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    	parser.add_argument('--n', type=int, default=10,
			help='number of nodes in the graph')
        parser.add_argument('--p', type=float, default=0.1,
                        help='probability')
	params = parser.parse_args()
	return params
	
if __name__ == "__main__":
	params = getparams()
	create(params.n, params.p)

	

