import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import glob
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.layout import *
from random import choice
from collections import defaultdict
		
def draw_graph(G, pos):
	
	pos = graphviz_layout(G)


	#nx.draw_networkx(G, pos=nx.spring_layout(G),with_labels= True)
	#nx.draw_networkx(G, pos=circular_layout(G),with_labels= True)
        #nx.draw_networkx(G, pos=shell_layout(G), with_labels=True)
        #nx.draw_networkx(G, pos=spectral_layout(G), with_labels=True)
        #nx.draw_networkx(G, pos=fruchterman_reingold_layout(G), with_labels=True)
        #f = plt.figure()
        nx.draw(G, pos=pos, with_labels=True,ax=f.add_subplot(111))

        #rescale_layout
        #print G.nodes()
	#print G.neighbors(0)
	plt.axis('off')
        #plt.title(title)
	#f.savefig(sys.argv[3])
        
        plt.show()
	return G
	
if __name__ == "__main__":
	
        #path = sys.argv[1]+"graph*"
        #path = [sys.argv[1]+"1.txt",sys.argv[1]+"2.txt",sys.argv[1]+"3.txt",sys.argv[1]+"4.txt" ]
        path = sys.argv[1]
        #with open(sys.argv[1]+"ll.txt") as f:
        count = 0
        #for fname in glob.glob(path):
        for fname in [path]:
            print fname
            f = open(fname, 'rb')
            G=nx.read_edgelist(f, nodetype=int)
            degree_sequence=list(nx.degree(G).values())
            print degree_sequence
            n = max(G.nodes())+1
            #print n
            pos = defaultdict()
            row = 0 
            col = 0
            for i in range(n):
                pos[i] = (row, col)
                row += 1
                if (i+1)%5 == 0:
                    col+=1
                    row = 0
                #row+=1
            draw_graph(G,pos)
            f.close()
            count +=1
            #break

	

