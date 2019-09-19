import collections
import networkx as nx
from collections import defaultdict
import numpy as np

def breadth_first_search(graph, degree, root): 
        visited, queue = set(), collections.deque([root])
        #degree = graph.degree()
        #print "Degree", degree
        bfs_nodes = [] 
        bfs_edges = []
        #print "Graph", graph
        while queue: 
            vertex = queue.popleft()
            bfs_nodes.append(vertex)
            if degree[vertex] > 0:
                neighbourlist = np.random.choice(graph[vertex], len(graph[vertex]), replace = False)
                for neighbour in neighbourlist: 
                    if neighbour not in visited: 
                        degree[vertex] -= 1
                        degree[neighbour] -=1
                        if degree[neighbour] == 0:
                            visited.add(neighbour)
                        if degree[vertex] == 0:
                            visited.add(vertex)
                        bfs_edges.append((vertex, neighbour))
                        if degree[neighbour] > 0:
                            queue.append(neighbour)
        return bfs_nodes, bfs_edges

def get_degree_distribution(deg):
    total_deg = sum(deg.values())
    degree_dist = []
    for i in range(len(deg.keys())):
        degree_dist.append(deg[i]*1.0/total_deg)
    return degree_dist


def get_edge_list_BFS(A, G, n, m, choice=None):
    #A = np.array(nx.adjacency_matrix(G).todense())
    adjdict = defaultdict()
    for node in G.nodes():
        adjdict[node] = np.nonzero(A[node])[0].tolist()
    degree_dist = get_degree_distribution(G.degree())
    max_deg = max(degree_dist)
    if choice == 'max':
        candidate_nodes = [x for x in G.nodes() if degree_dist[x] == max_deg]
        #np.argmax(degree_dist, axis=0).tolist()
        nodes = np.random.choice(candidate_nodes, min(n, len(candidate_nodes)), replace=False)
    else:
        nodes = np.random.choice(G.nodes(), n, p=degree_dist, replace=False)
    bfs_edge_list = []
    for node in nodes:
        for i in range(m):
            bfs_n, bfs_e = breadth_first_search(adjdict, G.degree(), node)
            bfs_edge_list.append(bfs_e)
            print("Debug BFS", bfs_e)
    return bfs_edge_list

if __name__ == '__main__':
    adjdict = defaultdict()
    G = nx.lollipop_graph(5, 3)
    A = np.array(nx.adjacency_matrix(G).todense())
    for node in G.nodes():
        adjdict[node] = np.nonzero(A[node])[0].tolist()
    degree_dist = get_degree_distribution(G.degree())
    nodes = np.random.choice(G.nodes(),3, p= degree_dist, replace=False)
    nodes = [4, 1, 5]
    for node in nodes:
        bfs_n, bfs_e = breadth_first_search(adjdict, G.degree(), node)
        print bfs_e, len(bfs_e)
    print get_edge_list_BFS(A,2,"max")
