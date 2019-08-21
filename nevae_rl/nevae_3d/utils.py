import tensorflow as tf
import os
import pickle
from numpy import *
import numpy as np
import networkx as nx
#import tensorflow as tf
import ast
import scipy
from numpy.linalg import svd, qr, norm
import glob
from collections import defaultdict
from math import log
from edge_sampling import *
import Queue as Q
from rdkit.Chem.Descriptors import qed
from checkvalidity import *
import copy
from operator import itemgetter
import subprocess
#import queue
#import sascorer
#from convert_to_nx import *

def get_coordinates(coord_mu, coord_sigma, adj, n=40):
    coordinates = defaultdict(list)
    q = Q.Queue(maxsize=n)
    q.put(0)
    coord = []
    while not q.empty():
        
        u = q.get()
        if len(coordinates[u]) == 0:
            predict_coord_u = np.random.multivariate_normal(coord_mu[u], coord_sigma[u])
            coordinates[u] = predict_coord_u
        predict_coord_u = coordinates[u]

        for i in range(n):
            if adj[u][i] == 1:
                adj[i][u] = 0
                for x in range(50):
                    predict_coord_i = np.random.multivariate_normal(coord_mu[i], coord_sigma[i])
                    dist = np.linalg.norm(np.array(predict_coord_u) - np.array(predict_coord_i)) 
                    if dist >= 1.0 and dist<=1.52:
                        print "X:", x
                        break
                coordinates[i] = predict_coord_i
                q.put(i)
    for i in range(n):
        coord.append(coordinates[i])

    return np.stack(coord)

def neighbor(z_coord, weight, z_dim):
	z_coord_modified = []
	for i in range(len(z_coord)):
		z_i = np.reshape(z_coord[i], (1, z_dim))
		temp = np.zeros(z_dim)
		for j in range(len(z_coord)):
			z_neighbor = np.reshape(z_coord[j], -1)
			#print("Debug weight", weight[i][j]* z_neighbor)
			temp = np.add(temp, weight[i][j] * z_neighbor)
		#print("temp_size", temp)
		z_coord_modified.append(np.concatenate((z_i,np.reshape(temp, (1,z_dim))), axis=1))
	return np.reshape(z_coord_modified, (len(z_coord),2 *z_dim))

def compute_cost_energy(weight, coords, features):
	filename = "/NL/random-graphs/work/gaussian/data/temp_2.gjf"
	f = open(filename, "w")
	#f.write(+"\n")
        print "Weriting t file"
        G = nx.from_numpy_matrix(np.matrix(weight))
	
        indices = np.argsort(features)
        indices = indices[::-1]
        
        adj = G.adjacency_list()
	count = 1
	f.write("%mem=16GB \n")
	f.write("%nprocs=6 \n")
	f.write("%nosave \n")
	
	
	#f.write("# b3lyp/6-311g(d) scf=(xqc, maxcycles=900)\n\n")
	f.write("# b3lyp/6-311g(d)\n\n")
	#f.write("# b3lyp/6-311g(d) geom=connectivity scf=(xqc,maxcycles=900)\n\n")
	f.write("Title Card Required\n\n")
	f.write("0 1\n")
        
        print "Debug ", indices, features
        for i in indices:
	#for node in G.nodes():
                node = G.nodes()[i]        
		if features[node] == 0:
                    f.write(" H                "+str(float("{0:.8f}".format(coords[node][0])))+"    "+str(float("{0:.8f}".format(coords[node][1])))+"    "+str(float("{0:.8f}".format(coords[node][2])))+"\n")


		if features[node] == 1:
                    f.write(" O                "+str(float("{0:.8f}".format(coords[node][0])))+"    "+str(float("{0:.8f}".format(coords[node][1])))+"    "+str(float("{0:.8f}".format(coords[node][2])))+"\n")

		if (features[node] == 2 or features[node] == 4):
                    f.write(" N                "+str(float("{0:.8f}".format(coords[node][0])))+"    "+str(float("{0:.8f}".format(coords[node][1])))+"    "+str(float("{0:.8f}".format(coords[node][2])))+"\n")

		if features[node] == 3:
                    f.write(" C                "+str(float("{0:.8f}".format(coords[node][0])))+"    "+str(float("{0:.8f}".format(coords[node][1])))+"    "+str(float("{0:.8f}".format(coords[node][2])))+"\n")
	
        f.write("\n")
	
	#'''
        for i in indices:
        #for el in adj:
                el = adj[i]
		str_w = " " + str(count)+" "
		for x in el:
                        x1 = np.where(indices == x)
			print x1[0][0]
                        if x1[0][0] > (count-1):
				w = G[i][x]["weight"] 
				str_w += str(x1[0][0]+1)+" "+str(w)+ " "
		f.write(str_w + "\n")
		count += 1

	while count < len(G.nodes()):
		f.write(" "+str(count) + "\n")
        #'''
        f.close()
        return getenergy("temp_2.gjf")

def getenergy(name, args):
        my_env = os.environ.copy()
        my_env["GAUSS_SCRDIR"]=args.gauss_src
        my_env["g09root"]=args.gauss_src
        my_env["GAUSS_EXEDIR"]=args.gauss_src
	filename = args.src_data + name
        print filename	
        #prog = subprocess.Popen(["/NL/random-graphs/work/gaussian/g09/g09 ",filename], env=my_env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        #out, err = prog.communicate()
        #print "Wait", prog.wait()
        #print "Debug out err", out, err
	var = os.system(args.gauss_src + "g09 " + filename)
	logfile = args.gauss_data + name.replace("gjf", "log")
	
	p = os.popen("grep \"SCF Done\" "+ logfile, "r")
	grepped_string = p.readline()
        print grepped_string
	value = grepped_string[25:25+9]
        try:
            ret_val = 1000.0 + float(value)
        except:
            ret_val = 2000.0
	return ret_val

def get_weighted_edges_connected(indicator, prob, edge_mask, w_edge, n_edges, node_list, degree_mat, start):
            i = 0
            candidate_edges = []
            q = Q.PriorityQueue()
            q.put((0, start))
            visited = np.zeros(len(node_list))
            for i in range(len(node_list)):
                for j in range(len(node_list)):
                   if node_list[i] == 1 and node_list[j] == 1:
                       edge_mask[i][j] = 0
            i = 0
            try:
             while not q.empty():
                start = q.get()[1]
                visited[start] = 1
                d = degree_mat[start]
                while d < node_list[start]:
                    list_edges = get_candidate_neighbor_edges(start, len(node_list))
                    p = normalise_h1(prob, w_edge, indicator.shape[1], indicator, edge_mask, start)
                    candidate_edges.extend([list_edges[k] for k in
                                        np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])
                    (u, v, w) = candidate_edges[i]
                    degree_mat[u] += w
                    degree_mat[v] += w

                    edge_mask[u][v] = 0
                    edge_mask[v][u] = 0

                    if (node_list[u] - degree_mat[u]) == 0:
                        indicator[u][0] = 0
                    if (node_list[u] - degree_mat[u]) <= 1:
                        indicator[u][1] = 0
                    if (node_list[u] - degree_mat[u]) <= 2:
                        indicator[u][2] = 0

                    if (node_list[v] - degree_mat[v]) == 0:
                        indicator[v][0] = 0
                    if (node_list[v] - degree_mat[v]) <= 1:
                        indicator[v][1] = 0
                    if (node_list[v] - degree_mat[v]) <= 2:
                        indicator[v][2] = 0
                    i += 1
                    d += w
                    if u == start:
                        if visited[v] == 0:
                            #q.put((node_list[v], v))
                            q.put((4 - node_list[v], v))
                    else:
                        if visited[u] == 0:
                            #q.put((node_list[u], u))
                            q.put((4 - node_list[u], u))
                    #print("Candidate_edges", start, d, node_list[start],len(candidate_edges), q.empty())
            except:
                candidate_edges = []
            # if the alignment was no possible
            if len(candidate_edges) < n_edges - 2:
                print("Some issue", candidate_edges, len(candidate_edges))
                candidate_edges = []
            return candidate_edges


def get_masked_candidate(n, list_edges_original, known_edges, prob, w_edge, num_edges, indicator=[], degree_original=[], atom_list=[], bin_dim=3):
        #list_edges_original = copy.deepcopy(list_edges)
        prob = np.reshape(prob, (n,n))
        w_edge = np.reshape(w_edge, (n, n, bin_dim))
        
        count  = 0
        structure_list = defaultdict(int)
        candidate_edges_list = []
        while (count < 1000):
            applyrules = False
            
            list_edges = copy.deepcopy(list_edges_original)
            known_edges_local = copy.deepcopy(known_edges)
            if len(indicator) == 0 :
                print("Debug indi new assign")
                indicator_local = np.ones([n, bin_dim])
            indicator_local = copy.deepcopy(indicator)
            p, list_edges, w = normalise(prob, w_edge, n, bin_dim, known_edges_local, list_edges, indicator_local)
            #print "Debug p", len(p), len(list_edges)
            candidate_edges = [list_edges[k] for k in
                               np.random.choice(range(len(list_edges)), [1], p=p, replace=False)]

            if len(degree_original) == 0:
                print("Debug degree new assign")
                degree = np.zeros([self.n])
            degree = copy.deepcopy(degree_original)
            G = None
            saturation = 0
            try:
                for i1 in range(num_edges - 1):
                    (u, v, w) = candidate_edges[i1]
                    if (u, v, 1) in list_edges:
                        list_edges.remove((u,v,1))
                    if (u, v, 2) in list_edges:
                        list_edges.remove((u,v,2))
                    if (u, v, 3) in list_edges:
                        list_edges.remove((u,v,3))
                    degree[u] += w
                    degree[v] += w
                    #print "Degree debug", degree[u], degree[v]
                    if (atom_list[u] - degree[u]) == 0:
                        indicator_local[u][0] = 0
                    if (atom_list[u] - degree[u]) <= 1:
                        indicator_local[u][1] = 0
                    if (atom_list[u] - degree[u]) <=2:
                        indicator_local[u][2] = 0

                    if (atom_list[v] - degree[v]) == 0:
                        indicator_local[v][0] = 0
                    if (atom_list[v] - degree[v]) <= 1:
                        indicator_local[v][1] = 0
                    if (atom_list[v] - degree[v]) <= 2:
                        indicator_local[v][2] = 0
                
                    # there will ne bo bridge
                    known_edges_local.extend(candidate_edges)
                    
                    p, list_edges, w = normalise(prob, w_edge, n, bin_dim, known_edges_local, list_edges, indicator_local)
                    candidate_edges.extend([list_edges[k] for k in
                                       np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])
                count += 1 
            except Exception as e: 
                    #print e
                    candidate_edges_list.append(candidate_edges)
                    count += 1
                    continue
            
            candidate_edges_list.append(candidate_edges)
            #structure_list[' '.join([str(u)+ '-'+str(v)+'-'+str(w) for (u,v,w) in sorted(candidate_edges)])] += 1
            count += 1

        #return the element which has been sampled maximum time
        #print "indicator before return", indicator, candidate_edges
        len_array = np.array([len(c) for c in candidate_edges_list])
        index = np.argmax(len_array)
        '''
	for candidate_edges in candidate_edges_list:
		structure_list[' '.join([str(u)+ '-'+str(v)+'-'+str(w) for (u,v,w) in sorted(candidate_edges)])] += 1 
        print structure_list
	max_candidate = max(structure_list.iteritems(), key=itemgetter(1))[0]
        max_candidate_edges = [x.split("-") for x in max_candidate.split(" ")]
	return max_candidate_edges
	'''
	return candidate_edges_list
        #return candidate_edges_list[index]
        #return max(structure_list.iteritems(), key=itemgetter(1))[0]

def change(p, w, hnodes, nodes, bin_dim, degree, indicator):
    p_matrix = np.zeros((nodes - len(hnodes), nodes - len(hnodes)))
    w_matrix = np.zeros((nodes - len(hnodes), nodes - len(hnodes), bin_dim))
    indicator_new = np.zeros((nodes - len(hnodes), bin_dim))
    degree_new = np.zeros(nodes - len(hnodes))
    rest = list(set(range(nodes)) - set(hnodes)) 
    print("Debig dim", rest, len(rest), p_matrix.shape)
    k = 0
    for i in rest:
        l = 0
        degree_new[k] = degree[i]
        indicator_new[k] = indicator[i]
        for j in rest:
                p_matrix[k][l] = p[i][j]
                w_matrix[k][l] = w[i][j]
                l += 1
        k += 1 
    return (p_matrix, w_matrix, degree_new, indicator_new)

def normalise_accross_edges(dict1, indicator, edge_mask):
    p_list = []
    for (u,v,w) in dict1.keys():
        p = dict1[(u,v,w)]
        indi = np.multiply(indicator[u], indicator[v])
        p1 = p * edge_mask[u][v] * indicator[u][w-1]
        p_list.append(p)
    return list(p_list/sum(p_list))

def normalise_h2(prob, weight, bin_dim, indicator, edge_mask, list_edges):
    n = len(prob[0])
    temp = np.ones([n, n])
    p_rs = np.exp(np.minimum(prob, 10 * temp))

    
    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10 * temp))
    combined_problist = []
    problist = []
    candidate_list_edges = []
    for (u,v) in list_edges:
        for i in range(bin_dim):
            candidate_list_edges.append((u, v, i+1))
        problist.append(p_rs[u][v]*edge_mask[u][v])
        
        indi = np.multiply(indicator[u], indicator[v])
        denom = sum(np.multiply(w_rs[u][v], indi))
        if denom == 0:
            denom = 1
            #del problist[-1]
        w_rs[u][v] = np.multiply(w_rs[u][v], indi) / denom
        combined_problist.extend(p_rs[u][v] * edge_mask[u][v] * w_rs[u][v])

    combined_problist = np.array(combined_problist)
    #print("Debug utils", problist, combined_problist)
    return combined_problist / combined_problist.sum(), candidate_list_edges

def normalise_h1(prob, weight, bin_dim, indicator, edge_mask, node):
    n = len(prob[0])
    temp = np.ones([n, n])
    p_rs = np.exp(np.minimum(prob, 10 * temp))
   
    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10 * temp))
    combined_problist = []
    problist = []

    for j in range(n):
        if j != node:
            if j < node:
                problist.append(p_rs[j][node] * edge_mask[j][node])
                indi = np.multiply(indicator[node], indicator[j])
                denom = sum(np.multiply(w_rs[j][node], indi))
                if denom == 0:
                    denom = 1
                    del problist[-1]
                w_rs[j][node] = np.multiply(w_rs[node][j], indi) / denom
                combined_problist.extend(p_rs[j][node] * w_rs[j][node] * edge_mask[j][node])
            else:
                problist.append(p_rs[node][j] * edge_mask[node][j])
                indi = np.multiply(indicator[node], indicator[j])
                denom = sum(np.multiply(w_rs[node][j], indi))
                if denom == 0:
                    denom = 1
                    del problist[-1]
                w_rs[node][j] = np.multiply(w_rs[node][j], indi) / denom
                combined_problist.extend(p_rs[node][j] * w_rs[node][j] * edge_mask[j][node])
    problist = np.array(problist)
    combined_problist = np.array(combined_problist)
    

    print("Debug", combined_problist.sum(), problist.sum(), problist)
    #return combined_problist / problist.sum()
    return combined_problist / combined_problist.sum()

def normalise_h(prob, weight, bin_dim , indicator, edge_mask, indexlist):
    
    n = len(prob[0])
    temp = np.ones([n, n])
    p_rs = np.exp(np.minimum(np.multiply(prob, edge_mask), 10 * temp))
    
    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10* temp))
    combined_problist = []
   
    problist = []
    for i in indexlist:
        for j in range(i+1, n):
            problist.append(p_rs[i][j])
            indi = np.multiply(indicator[i], indicator[j])
            denom = sum(np.multiply(w_rs[i][j], indi))
            if denom == 0:
                denom = 1
                del problist[-1]
            w_rs[i][j] = np.multiply(w_rs[i][j], indi)/ denom
            combined_problist.extend(p_rs[i][j]*w_rs[i][j])
    problist = np.array(problist)
    
    return combined_problist/problist.sum()

def checkcycle(edge, G=None):
    if G == None:
        G=nx.Graph()
    (u, v, w) = edge
    G.add_edge(u, v, weight=w)
    #return (G, len(list(nx.simple_cycles(G))))
    return (G, len(nx.cycle_basis(G)))

def log_fact(k):
    dict_ = defaultdict(float)
    for i in range(k):
        dict_[i+1] = dict_[i] + log(i+1)
    return dict_ 

def normalise(prob, weight, n, bin_dim, seen_list, list_edges, indicator):
    n = len(prob[0])
    temp = np.ones([n, n])
    p_rs = np.exp(np.minimum(prob, 10 * temp))
    p_rs = p_rs/p_rs.sum()
    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10* temp))
    #w_rs = p_rs/p_rs.sum()
    combined_problist = []
    problist = []
    #print "Debug seen list", seen_list
    for (i, j, w) in list_edges:        
        problist.append(p_rs[i][j])
        indi = np.multiply(indicator[i], indicator[j])
        denom = sum(np.multiply(w_rs[i][j], indi))
        if denom == 0:
                denom = 1
        temp = np.multiply(w_rs[i][j], indi)/ denom
        #combined_problist.extend(p_rs[i][j]*w_rs[i][j])
        combined_problist.append(p_rs[i][j] * temp[w-1])
    problist = np.array(problist)
    combined_problist = np.array(combined_problist)
    return combined_problist/combined_problist.sum(), list_edges, w_rs

def get_candidate_edges(n):
    list_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # list_edges.append((i,j))
            list_edges.append((i, j, 1))
            list_edges.append((i, j, 2))
            list_edges.append((i, j, 3))
    return list_edges

def get_candidate_neighbor_edges(index, n):
    list_edges = []
    for j in range(n):
            if j == index:
                continue
            if j > index:
                #list_edges.append((index,j))
                list_edges.append((index, j, 1))
                list_edges.append((index, j, 2))
                list_edges.append((index, j, 3))
            else:
                #list_edges.append((j,index))
                list_edges.append((j, index, 1))
                list_edges.append((j, index, 2))
                list_edges.append((j, index, 3))
    
    return list_edges

def slerp(p0, p1, t):
    omega = arccos(dot(p0/norm(p0), p1/norm(p1)))
    so = sin(omega)
    if so == 0:
        return p0
    #print "Debug slerp", p0, p1, omega, so,  sin((1.0-t)*omega)/so,  sin((1.0-t)*omega)/so *np.array(p0)
    return sin((1.0-t)*omega) / so * np.array(p0) + sin(t*omega)/so * np.array(p1)

def lerp(p0, p1, t):
    return np.add(p0, t * np.subtract(p1,p0))

def degree(A):
    return np.zeros()


def construct_feed_dict(lr,dropout, k, n, d, decay, placeholders):
    # construct feed dictionary
    feed_dict = dict()


    #feed_dict.update({placeholders['features']: features})
    #feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['lr']: lr})
    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['decay']: decay})
    #feed_dict.update({placeholders['input']:np.zeros([k,n,d])})
    return feed_dict


def get_shape(tensor):
    '''return the shape of tensor as list'''
    return tensor.get_shape().as_list()

def basis(adj, atol=1e-13, rtol=0):
    A = degree(adj) - adj

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    q, r = qr(A)
    return q[:rank]

def print_vars(string):
    '''print variables in collection named string'''
    print("Collection name %s"%string)
    print("    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in tf.get_collection(string)]))

def get_basis(mat):
    basis = np.zeros(1,1)
    return basis

def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def pickle_load(path):
    '''Load the picke data from path'''
    with open(path, 'rb') as f:
        loaded_pickle = pickle.load(f)
    return loaded_pickle

def load_embeddings(fname, z_dim):
    embd = []
    with open(fname) as f:
        for line in f:
            embd.append(np.array(ast.literal_eval(line)).reshape((z_dim,1)))
    return embd

def load_data(filename, num=0, bin_dim=3):
    path = filename+"/*"
    adjlist = []
    featurelist = []
    weightlist = []
    weight_binlist = []
    edgelist = []
    hdelist = []
    filenumber = int(len(glob.glob(path)) * 0.2)
    neg_edgelist = []
    neg_indexlist = []

    for fname in sorted(glob.glob(path))[:filenumber]:
        f = open(fname, 'r')
        try:
            G=nx.read_edgelist(f, nodetype=int)
        except:
            f = open(fname, 'r')
            lines = f.read()
            linesnew = lines.replace('{', '{\'weight\':').split('\n')
            G=nx.parse_edgelist(linesnew, nodetype=int)
        
        f.close()
        n = num
        for i in range(n):
            if i not in G.nodes():
                G.add_node(i)

        # We assume there are only 4 types of atoms 
        degreemat = np.zeros((n, 4), dtype=np.float)
        count = np.zeros(4)

        for u in G.nodes():
            if G.degree(u) == 3 or G.degree(u) == 5:
                index = 2
            else:
                index = G.degree(u) -1
            degreemat[int(u)][index] = 1 
            #degreemat[int(u)][0] = (G.degree(u)*1.0)/(n-1)
            #count[G.degree(u)] += 1

        hde = (2 * count[3] + 2 + count[2] - count[0]) / 2
        hdelist.append(hde)
        #neg_edgelist = []
        #neg_indexlist = []
       
        e = len(G.edges())
        try:
            weight = np.array(nx.adjacency_matrix(G).todense())
            adj = np.zeros([n,n])
            weight_bin = np.zeros([e,bin_dim])
            edges = []
            count = 0
            pos_count = 0
            neg_edges = []
            neg_index = []
            for i in range(n):
                for j in range(i+1,n):
                    if weight[i][j]>0:
                        adj[i][j] = 1
                        weight_bin[pos_count][weight[i][j]-1] = 1
                        edges.append((i,j))
                        pos_count += 1
                    else:
                        neg_edges.append((i,j))
                        neg_index.append(count)
                    count += 1
            adjlist.append(adj)
            weightlist.append(weight)
            weight_binlist.append(weight_bin)
            featurelist.append(degreemat)
            edgelist.append(edges)
            neg_edgelist.append(neg_edges)
            neg_indexlist.append(neg_index)
        except:
            print("Error")
            continue
    
    return (adjlist, weightlist, weight_binlist, featurelist, edgelist, hdelist, neg_edgelist, neg_indexlist)

def pickle_save(content, path):
    '''Save the content on the path'''
    with open(path, 'wb') as f:
        pickle.dump(content, f)


def getedges(adj, n):
    edges = []
    for i in range(n):
        for j in range(n):
            if adj[i][j] > 0:
                edges.append((i,j, adj[i][j]))
    return edges

def get_weight_bins(n, bin_dim, G):
    weight_bin = np.zeros([n,n,bin_dim])
    
    for (u, v, w) in G.edges_iter(data='weight'):   
        weight_bin[u][v][w-1] = 1
        weight_bin[u][v][w-1] = 1
        
    return weight_bin 

def load_data_new(filename, num, node_sample, edge_sample, bin_dim=3, synth=False):
    path = filename+"/*"
    adjlist = []
    featurelist = []
    featurelist1 = []
    weightlist = []
    weight_binlist = []
    weight_bin_list1 = []
    edgelist = []
    smiles = []
    
    #filenumber = int(len(glob.glob(path)) * 0.1)
    
    filenumber = 1
    atom_list = []
    neg_edgelist = []
    
    for fname in sorted(glob.glob(path))[: filenumber]:
        f = open(fname, 'r')
        try:
            G=nx.read_edgelist(f, nodetype=int)
        except:
            f = open(fname, 'r')
            lines = f.read()
            linesnew = lines.replace('{', '{\'weight\':').split('\n')
            G=nx.parse_edgelist(linesnew, nodetype=int)
        #''' 
        if not synth:
            if guess_correct_molecules(fname, 'temp.txt', num, 1):
                    m = Chem.MolFromMol2File('temp.txt')
                    if m != None:
                        smiles.append(Chem.MolToSmiles(m))
                    else:
                        smiles.append('')
                        continue
        #'''
        f.close()
        n = num
        for i in range(n):
            if i not in G.nodes():
                G.add_node(i)
        atom_list.append(G.degree())
        # We assume there are only 4 types of atoms 
        degreemat = np.zeros((n, 4), dtype=np.float)
        count = np.zeros(4)
        degree_1 = []
        #print("Debug degree", G.degree().values())
        #np.zeros(n)
        if synth:
            degreemat = np.zeros((n, 1), dtype=np.float)
            for u in G.nodes():
                degreemat[int(u)][0] = G.degree(u)
                degree_1.append(0)
        else:   
            for u in G.nodes():
                    if G.degree(u) == 3 or G.degree(u) >= 5:
                            index = 2
                    else:
                        index = G.degree(u) -1
                    degreemat[int(u)][index] = 1 
                    degree_1.append(index)
            
        e = len(G.edges())
        
        #try:
        weight = np.array(nx.adjacency_matrix(G).todense())
        adj = np.zeros([n,n])
        weight_bin_list = [] 
        count = 0
        pos_count = 0
        neg_edges = []

        #edge_list = get_edge_list_BFS(weight, G, node_sample, edge_sample,"max")
        edge_list = [G.edges()]
        for edges in edge_list:
                count_pos = 0
                weight_bin = []
                #weight_bin = np.zeros([n,n,bin_dim])
                #np.zeros([bin_dim])
                for (i, j) in edges:
                    temp = np.zeros([bin_dim])
                    temp[weight[i][j]-1] = 1
                    weight_bin.append(temp)
                    count_pos += 1
                weight_bin_list.append(weight_bin)
        weight_bin = np.zeros([n,n,bin_dim])
        
        for i in range(n):
            for j in range(n):
                if weight[i][j]>0:
                    adj[i][j] = 1
                    weight_bin[i][j][weight[i][j]-1] = 1
                neg_edges.append((i,j))
        weight_bin_list1.append(weight_bin)        
        #count += 1

        adjlist.append(adj)
        weightlist.append(weight)
        weight_binlist.append(weight_bin_list)
        featurelist.append(degreemat)
        featurelist1.append(degree_1)
        edgelist.append(edge_list)
        neg_edgelist.append(neg_edges)
    return (adjlist, weightlist, weight_binlist, weight_bin_list1, featurelist, edgelist, neg_edgelist, featurelist1, atom_list, smiles)
    #return (adjlist, weightlist, weight_binlist, weight_binlist1, featurelist, edgelist, neg_edgelist, featurelist1, degrees, smiles)

def get_weighted_edges_connected(indicator, prob, edge_mask, w_edge, n_edges, node_list, degree_mat, start):
            i = 0
            candidate_edges = []
            q = Q.PriorityQueue()
            q.put((0, start))
            visited = np.zeros(len(node_list))
            for i in range(len(node_list)):
                #if node_list[i] == 1:
                #    q.put((0, i))
                for j in range(len(node_list)):
                   if node_list[i] == 1 and node_list[j] == 1:
                       edge_mask[i][j] = 0
            i = 0
            try:
             while not q.empty():
                start = q.get()[1]
                visited[start] = 1
                d = degree_mat[start]
                while d < node_list[start]:
                    list_edges = get_candidate_neighbor_edges(start, len(node_list))
                    p = normalise_h1(prob, w_edge, indicator.shape[1], indicator, edge_mask, start)
                    candidate_edges.extend([list_edges[k] for k in
                                        np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])
                    (u, v, w) = candidate_edges[i]
                    degree_mat[u] += w
                    degree_mat[v] += w

                    edge_mask[u][v] = 0
                    edge_mask[v][u] = 0

                    if (node_list[u] - degree_mat[u]) == 0:
                        indicator[u][0] = 0
                    if (node_list[u] - degree_mat[u]) <= 1:
                        indicator[u][1] = 0
                    if (node_list[u] - degree_mat[u]) <= 2:
                        indicator[u][2] = 0

                    if (node_list[v] - degree_mat[v]) == 0:
                        indicator[v][0] = 0
                    if (node_list[v] - degree_mat[v]) <= 1:
                        indicator[v][1] = 0
                    if (node_list[v] - degree_mat[v]) <= 2:
                        indicator[v][2] = 0
                    
                    i += 1
                    d += w
                    
                    if u == start:
                        if visited[v] == 0:
                            #q.put((node_list[v], v))
                            q.put((4 - node_list[v], v))
                    else:
                        if visited[u] == 0:
                            #q.put((node_list[u], u))
                            q.put((4 - node_list[u], u))
                    #print("Candidate_edges", start, d, node_list[start],len(candidate_edges), q.empty())
            except:
                candidate_edges = []
            # if the alignment was no possible
            if len(candidate_edges) < n_edges - 2:
                print("Some issue", candidate_edges, len(candidate_edges))
                candidate_edges = []

            return candidate_edges

def load_data_from_pkl(hparams):
    G_list = pickle.load(open(hparams.graph_file))
    adjlist = []
    featurelist = []
    featurelist1 = []
    weightlist = []
    weight_binlist = []
    edgelist = []
    neg_edgelist = []
    smiles = [] 
    coordinates = []
    n = hparams.nodes
    bin_dim = hparams.bin_dim
    
    print "Len G_list", len(G_list)
    for G in G_list[26:27]:
	print "Debug G", G
	# degree can be 1, 2, 3, 4 plus the x,y,z coordinates
	# 
	degreemat = np.zeros((n, 4))
	degree = []
        adj = np.array(nx.adjacency_matrix(G).todense())
        weight = np.zeros((n, n))
	weight_bin_comblist = [] 
	#np.zeros((len(G.edges()), bin_dim))
        neg_edges = nx.complement(G).edges()
        edge_list = get_edge_list_BFS(adj, G, hparams.node_sample, hparams.bfs_sample, "max")
        double_count = False
        #print "Debug edge_list", edge_list[0]
        for (i, j) in G.edges():
		    bond_type = G.get_edge_data(i,j)["bond_type"]
                    if bond_type == rdkit.Chem.rdchem.BondType.AROMATIC:
			if double_count == False:
				bond = 1
				double_count = True
			else:
				bond = 2
				double_count = False
		    if bond_type == rdkit.Chem.rdchem.BondType.SINGLE:
			bond = 1
		    if bond_type == rdkit.Chem.rdchem.BondType.DOUBLE:
 			bond = 2
		    if bond_type == rdkit.Chem.rdchem.BondType.TRIPLE:
                        bond = 3
		    weight[i][j] = bond
		    weight[j][i] = bond

	for edges in edge_list:
		weight_bin = []
                for (i, j) in edges:
                    temp = np.zeros(3)
                    temp[int(weight[i][j])-1] = 1
                    weight_bin.append(temp)
		weight_bin_comblist.append(weight_bin)	
	for u in G.nodes():
	    #(x,y,z) = coordinates[u]
	    deg = int(sum(weight[u]))
            if deg == 3 or deg >= 5:
                index = 2
            else:
                index = deg -1
	    print u,deg
            degreemat[int(u)][index] = 1
            degree.append(index)


        adjlist.append(adj)
        weightlist.append(weight)
        weight_binlist.append(weight_bin_comblist)
        featurelist.append(degreemat)
        featurelist1.append(degree)
        edgelist.append(edge_list)
        neg_edgelist.append(neg_edges)
	coordinates.append(nx.get_node_attributes(G, 'coord').values())

    return (adjlist, weightlist, weight_binlist, featurelist, edgelist, neg_edgelist, featurelist1, coordinates)


def get_weighted_edges_connected1(indicator, prob, edge_mask, w_edge, n_edges, node_list, degree_mat, start):
            i = 0
            candidate_edges = []
            q = Q.Queue()
            q.put(start)
            p = []
            list_edges = []
            list_nodes = [0]
            
            visited = np.zeros(indicator.shape[0])
            while i < n_edges :
                #start = q.get()
                #visited[start] = 1
                if visited[start] == 0:
                    list_edges.extend(get_candidate_neighbor_edges(start, len(node_list)))
                    p_new = normalise_h1(prob, w_edge, indicator.shape[1], indicator, edge_mask, start)
                    p.extend(p_new.tolist())
                    #print("Debug p_new", p, list_edges)
                    dictionary = dict(zip(list_edges, p))
                    list_edges = dictionary.keys()  
                    p = dictionary.values()
                    p = list(p/sum(p))
                    visited[start] = 1
                else:
                    print("Else")
                    p = normalise_accross_edges(dictionary, indicator, edge_mask)
                #print("Debug", list_edges,p)
                print("Debug ", len(list_edges), p, i)
                candidate_edges.extend([list_edges[k] for k in
                                        np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])
                
                (u, v, w) = candidate_edges[i]
                degree_mat[u] += w
                degree_mat[v] += w

                edge_mask[u][v] = 0
                edge_mask[v][u] = 0

                if (node_list[u] - degree_mat[u]) == 0:
                        indicator[u][0] = 0
                if (node_list[u] - degree_mat[u]) <= 1:
                        indicator[u][1] = 0
                if (node_list[u] - degree_mat[u]) <= 2:
                        indicator[u][2] = 0

                if (node_list[v] - degree_mat[v]) == 0:
                        indicator[v][0] = 0
                if (node_list[v] - degree_mat[v]) <= 1:
                        indicator[v][1] = 0
                if (node_list[v] - degree_mat[v]) <= 2:
                        indicator[v][2] = 0
                i += 1
                if visited[u] == 0:
                    start = u
                else:
                    start = v
                print("Candidate", candidate_edges, start)
            return candidate_edges

def get_weighted_edges_connected2(indicator, prob, edge_mask, w_edge, n_edges, node_list, degree_mat, start):
            i = 0
            candidate_edges = []
            q = Q.Queue()
            q.put(start)
            p = []
            list_edges = []
            list_nodes = [0]
            n = indicator.shape[0]

            list_edges = get_candidate_edges(n)
            try: 
             while i < n_edges :
                
                p = normalise_h(prob, w_edge, 3, indicator, edge_mask, range(n))
                
                if np.count_nonzero(p) == 0:
                    break
                print("Debug p",np.count_nonzero(p), len(list_edges))
                candidate_edges.extend([list_edges[k] for k in
                                        np.random.choice(range(len(list_edges)), [1], p=p, replace=False)])
                
                (u, v, w) = candidate_edges[i]
                degree_mat[u] += w
                degree_mat[v] += w

                edge_mask[u][v] = 0
                edge_mask[v][u] = 0

                if (node_list[u] - degree_mat[u]) == 0:
                        indicator[u][0] = 0
                if (node_list[u] - degree_mat[u]) <= 1:
                        indicator[u][1] = 0
                if (node_list[u] - degree_mat[u]) <= 2:
                        indicator[u][2] = 0

                if (node_list[v] - degree_mat[v]) == 0:
                        indicator[v][0] = 0
                if (node_list[v] - degree_mat[v]) <= 1:
                        indicator[v][1] = 0
                if (node_list[v] - degree_mat[v]) <= 2:
                        indicator[v][2] = 0
                i += 1
                #print("Debug candidate nodes", len(candidate_edges))
            except:
                 candiadate_edges = []
            return candidate_edges

def compute_cost_dia(G):
	if nx.is_connected(G):
		#return nx.diameter(G) - 11
		prop = nx.diameter(G) - 7
		if prop < 0:
			prop = 0
		return prop * 1.0 / 10000000000000
		#return 
		#return (20.0 - (7.0 - nx.diameter(G)))/10000000000000
	else:
		return 100

def compute_cost_qed(G, writefile="temp.txt"):
    qed_val = 2.0
    if guess_correct_molecules_from_graph(G, writefile):
        m1 = Chem.MolFromMol2File(writefile)
        if m1 != None:
		qed_val = 1.0 - qed(m1)
	else:
		print "Error: None"
    else:
	print "Error: wrong molecule"

    return qed_val



def compute_cost(G, writefile="temp.txt"):
    cost = 0.0
    sas = ""
    logP = ""
    cycle_score = ""
    #m1 = nx_to_mol(G)
    if guess_correct_molecules_from_graph(G, writefile):
        m1 = Chem.MolFromMol2File(writefile)
        if m1 != None:
            s = Chem.MolToSmiles(m1)
            sas = -sascorer.calculateScore(m1)
            logP = Descriptors.MolLogP(m1)
            cycle_list = nx.cycle_basis(G)

            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0.0
            else:
                cycle_length = cycle_length - 6.0
            cycle_score = -cycle_length
            cost = sas + logP + cycle_score
        else:
            print "Error: m1 is NONE"
            cost = ""
    else:
        print "Error: gues correct molecule"
        cost = ""
    # we want to define this property value such that low vales are better
    if cost != "":
        cost = 10.00 - cost
    return (sas, logP, cycle_score)
    #return cost
    #return (10.00 - cost)

def get_masked_candidate_new(prob, w_edge, n_edges, labels, indicator, edge_mask, degree):

        n = prob.shape[0]
        list_edges = get_candidate_edges(n)
        max_node = np.argmax(labels)
        #max_node = np.argmin(labels)
        #indicator = np.ones([self.n, self.bin_dim])
        #edge_mask = np.ones([self.n, self.n])
        #degree = np.zeros(self.n)
        candidate_edges= get_weighted_edges_connected2(indicator, prob, edge_mask, w_edge, n_edges, labels, degree, max_node)
        candidate_edges_new = []
        for (u, v, w) in candidate_edges:
            if u < v:
                candidate_edges_new.append(str(u) + ' ' + str(v) + ' ' + "{'weight':"+str(w)+"}")
            else:
                candidate_edges_new.append(str(v) + ' ' + str(u) + ' ' + "{'weight':"+str(w)+"}")
        print("Candidate_new", candidate_edges_new)
        return candidate_edges_new
