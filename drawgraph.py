import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import glob
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.layout import *
from random import choice
from collections import defaultdict
		
def draw_graph(G, name, pos):
        pos = graphviz_layout(G)
        nx.draw_networkx(G, node_size=400 ,pos=pos,with_labels= True)
        #nx.draw_networkx(G, pos=circular_layout(G),with_labels= True)
        plt.axis('off')
        #plt.title(title)
        plt.savefig(sys.argv[2]+name+'.pdf')
        plt.gcf().clear()

if __name__ == "__main__":
	
            path = sys.argv[1]+'*'
        
            for fname in sorted(glob.glob(path)):
                print fname
                f = open(fname, 'rb')
                G=nx.read_edgelist(f, nodetype=int)
                # = np.argsort(degree_seq)
                #getlikelihood(G, n, m)
                n= 32
                #len(G.nodes())
                
                pos = defaultdict()
                row = 0
                col = 0
                for i in range(n):
                    pos[i] = (row, col)
                    row += 1
                    if (i+1)%4 == 0:
                        col+=1
                        row = 0

                name = fname.split('/')[-1].split('.')[0]
                draw_graph(G,name,pos)

	

