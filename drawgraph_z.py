import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import glob
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.layout import *
from random import choice
from collections import defaultdict
import ast

def draw_graph(G):
        pos = graphviz_layout(G)
        nx.draw_networkx(G, node_size=600 ,pos=pos,with_labels= True)
        #nx.draw_networkx(G, pos=circular_layout(G),with_labels= True)
        plt.axis('off')
        #plt.title(title)
        plt.savefig(sys.argv[2]+'fig.png')
        #plt.show()

def getavg(path):
        avg_val = []
        for fname in sorted(glob.glob(path)):
                print fname
                f = open(fname)
                avg_x = 0.0
                avg_y = 0.0
                count = 0
                for line in f:
                    coordinates = ast.literal_eval(line)
                    avg_x += coordinates[0]
                    avg_y += coordinates[1]
                    count += 1
                avg_x /= count
                avg_y /= count
                avg_val.append((avg_x, avg_y))
        return avg_val 
def drawgraph(train, test):
    unzipped = zip(*train)
    plt.scatter(unzipped[0], unzipped[1])
    #n = len(train)
    #for i in range(n):
    #        plt.annotate(str(i), (train[i][0],train[i][1]))
    plt.xlim(-0.002, 0.002)
    plt.ylim(-0.0025, 0.0025)
    plt.savefig('test_z.png')
    
    #plt.show()

if __name__ == "__main__":
	
            path = sys.argv[1] + 'train*.txt'
            
            train_avg = getavg(path)
            print train_avg

            path = sys.argv[1] + 'test*.txt'

            test_avg = getavg(path)
            drawgraph(train_avg, test_avg)

	

