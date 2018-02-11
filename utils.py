import os
import pickle
from numpy import *
import numpy as np
import networkx as nx
import tensorflow as tf
import ast
import scipy
from numpy.linalg import svd, qr, norm
import glob



def normalise(prob, weight, n, bin_dim, seen_list, list_edges, indicator):
    #'''
    #print "Debug", np.minimum(prob, np.zeros([n, n]).fill(10.0))
    temp = np.ones([n, n])
    #print "Debug temp", temp
    #temp.fill(10.0)
    #print "Debug temp", np.minimum(prob, 10 * temp)

    p_rs = np.exp(np.minimum(prob, 10 * temp))
    p_rs = p_rs/p_rs.sum()
    temp = np.ones([n, n, bin_dim])
    w_rs = np.exp(np.minimum(weight, 10* temp))
    #w_rs = p_rs/p_rs.sum()
    combined_problist = []
    problist = []
    for i in range(n):
        for j in range(i+1, n):
            if (i,j,1) in seen_list or (i,j,2) in seen_list or (i,j,3) in seen_list:
                if (i, j, 1) in list_edges:
                    list_edges.remove((i, j, 1))
                if (i, j, 2) in list_edges:
                    list_edges.remove((i, j, 2))
                if (i, j, 3) in list_edges:
                    list_edges.remove((i, j, 3))
                continue

            problist.append(p_rs[i][j])
            indi = np.multiply(indicator[i], indicator[j])
            denom = sum(np.multiply(w_rs[i][j], indi))
            if denom == 0:
                denom = 1
                del problist[-1]
            w_rs[i][j] = np.multiply(w_rs[i][j], indi)/ denom
            combined_problist.extend(p_rs[i][j]*w_rs[i][j])
    problist = np.array(problist)
    #return problist/problist.sum(), list_edges, w_rs
    return combined_problist/problist.sum(), list_edges, w_rs
    #'''
    p_rs = prob/prob.sum()
    
    #p_rs = prob/prob.sum(axis=0)[:,None] 
    #p_new_rs = np.zeros([n,n,bin_dim])
    #w_rs = np.zeros([n, n, bin_dim])
    w_rs = weight
    problist = []
    negval = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            if (i,j,1) in seen_list or (i,j,2) in seen_list or (i,j,3) in seen_list:
                if (i, j, 1) in list_edges:
                    list_edges.remove((i, j, 1))
                if (i, j, 2) in list_edges:
                    list_edges.remove((i, j, 2))
                if (i, j, 3) in list_edges:
                    list_edges.remove((i, j, 3))
                continue
            #'''
            #w_rs[i][j] = weight[i][j]/ sum(weight[i][j])
            #w_rs[i][j] = np.exp(weight[i][j])/ sum(np.exp(weight[i][j]))
            #p_new_rs[i][j] = p_rs[i][j] * w_rs[i][j]
            
            probtemp = np.multiply(np.exp(np.minimum(p_rs[i][j]* w_rs[i][j], [10.0, 10.0, 10.0])), np.multiply(indicator[i], indicator[j]))
            #print("DEBUG proptemp", probtemp.sum())
            if probtemp.sum() > 0 :
                problist.extend(probtemp/ probtemp.sum())
            else:
                problist.extend(probtemp)
            negval += p_rs[i][j] * w_rs[i][j][0]
            #'''
    #print len(problist), negval
    #prob = np.triu(p_new_rs,1)
    #problist.append(negval)
    
    problist = np.array(problist)
    #print problist, problist.sum()
    #print problist.sum()
    return problist/problist.sum(), list_edges


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
    """Estimate the basis of a matrix.


    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    b : ndarray
        The basis of the columnspace of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
    """

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

def load_embeddings(fname):
    embd = []
    with open(fname) as f:
        for line in f:
            embd.append(ast.literal_eval(line))
    return embd

def load_data(filename, num=0, bin_dim=3):
    path = filename+"/*"
    adjlist = []
    featurelist = []
    weightlist = []
    weight_binlist = []
    edgelist = []
    filenumber = int(len(glob.glob(path)) * 1)
    
    for fname in sorted(glob.glob(path))[:filenumber]:
        f = open(fname, 'r')
        try:
            G=nx.read_edgelist(f, nodetype=int)
        except:
            print "Except"
            continue
        f.close()
        n = num
        for i in range(n):
            if i not in G.nodes():
                G.add_node(i)
        degreemat = np.zeros((n,1), dtype=np.float)

        for u in G.nodes():
            degreemat[int(u)][0] = (G.degree(u)*1.0)/(n-1)

        try:
            weight = np.array(nx.adjacency_matrix(G).todense())
            adj = np.zeros([n,n])
            weight_bin = np.zeros([n,n,bin_dim])
            edges = []
            for i in range(n):
                for j in range(n):
                    if weight[i][j]>0:
                        adj[i][j] = 1
                        weight_bin[i][j][weight[i][j]-1] = 1
                        if j > i:
                            edges.append((i,j,weight[i][j]))
            adjlist.append(adj)
            weightlist.append(weight)
            weight_binlist.append(weight_bin)
            featurelist.append(degreemat)
            edgelist.append(edges)
        except:
            continue
    return (adjlist, weightlist, weight_binlist, featurelist, edgelist)

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
