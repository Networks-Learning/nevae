
import pickle
import gzip
import sys
sys.path.insert(0, '../')
from checkvalidity import *
from sample import *

def decode_from_latent_space(latent_points, model, hparams):

    final_smiles = []
    valid = 0
    
    for point in latent_points:
        embedding = np.reshape(point,(hparams.nodes,hpamas.z_dim,1))
        valid_smiles = []
        valid_smiles = []
        i = 0
        while(i < 1000): 
            model.sample_graph_posterior_new(hparams, placeholders, adj[0], features[0], weight_bin[0], weight[0], embedding)
            i +=1

            readfile = hparams.sample_file+"temp_new.txt"
            #for readfile in sorted(glob.glob(hparams.sample_file)):
            #if "ll.txt" in readfile:
            #continue

            if guess_correct_molecules(readfile, 'test.mol2', hparams.nodes, 1):
                #total += 1
                m1 = Chem.MolFromMol2File('test.mol2')
                if m1 != None:
                      valid_smiles.append(Chem.MolToSmiles(m1))
                      valid +=1
        final_smiles.append(valid_smiles)
   
    return final_smiles 
    
    
# We define the functions used to load and save objects

def save_object(obj, filename):

    """
    Function that saves an object to a file using pickle
    """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):

    """
    Function that loads an object from a file using pickle
    """

    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()

    return ret

from sparse_gp import SparseGP

import scipy.stats    as sps

import numpy as np

# We load the random seed

random_seed = int(np.loadtxt('data/random_seed.txt'))
np.random.seed(random_seed)

# We load the data
X = np.loadtxt('data/latent_features.txt')
y = -np.loadtxt('data/targets.txt')

#X = np.loadtxt('../../latent_features_and_targets_character/latent_faetures.txt')
#y = -np.loadtxt('../../latent_features_and_targets_character/targets.txt')
y = y.reshape((-1, 1))

n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]


np.random.seed(random_seed)
FLAGS = None

placeholders = {
    'dropout': tf.placeholder_with_default(0., shape=()),
    'lr': tf.placeholder_with_default(0., shape=()),
    'decay': tf.placeholder_with_default(0., shape=())
}

iteration = 0
#loading model
parser = argparse.ArgumentParser()
add_arguments(parser)
FLAGS, unparsed = parser.parse_known_args()
hparams = create_hparams(FLAGS)
                    
# loading the data from a file
adj, weight, weight_bin, features, edges, hde = load_data(hparams.graph_file, hparams.nodes, hparams.bin_dim)
num_nodes = adj[0].shape[0]
num_features = features[0].shape[1]
e = max([len(edge) for edge in edges])
n_f = len(features[0][0])
log_fact_k = log_fact(e) 
model = VAEG(hparams, placeholders, hparams.nodes, n_f, edges, log_fact_k, hde)
model.restore(hparams.out_dir)

while iteration < 1:

    # We fit the GP

    np.random.seed(iteration * random_seed)
    M = 50
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
        y_test, minibatch_size = 10 * M, max_iterations = 50, learning_rate = 0.0005)

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print 'Test RMSE: ', error
    print 'Test ll: ', testll

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print 'Train RMSE: ', error
    print 'Train ll: ', trainll

    # We load the decoder to obtain the molecules

    from rdkit.Chem import MolFromSmiles, MolToSmiles
    from rdkit.Chem import Draw
    import image
    import copy
    import time

    import sys

    # We pick the next 50 inputs

    next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))

    smiles_list = decode_from_latent_space(next_inputs, model, hparams)

    from rdkit.Chem import Descriptors
    from rdkit.Chem import MolFromSmiles, MolToSmiles

    new_features = next_inputs
    #save_object(valid_smiles_final, "results/valid_smiles{}.dat".format(iteration))

    logP_values = np.loadtxt('data/logP_values.txt')
    SA_scores = np.loadtxt('data/SA_scores.txt')
    cycle_scores = np.loadtxt('data/cycle_scores.txt')

    SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
    cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

    targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized

    import sascorer
    import networkx as nx
    from rdkit.Chem import rdmolops

    scores = []
    valid_smiles_final_new = []

    #for i in range(len(valid_smiles_final)):
    for k in range(len(smiles_list)):
        valid_smiles_final = smiles_list[k]
        score = -1e10
        for i in range(len(valid_smiles_final)):
          if valid_smiles_final[ i ] is not None:
            current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles_final[ i ]))
            current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles_final[ i ]))
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles_final[ i ]))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([ len(j) for j in cycle_list ])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6

            current_cycle_score = -cycle_length
         
            current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
            current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
            current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)
            
            current_score = (current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized)
            if score < current_score:
                smile = valid_smiles_final[i]
                score = current_score
          else:
            score = -max(y)[ 0 ]
            smile = 'None'
        scores.append(-score)
        valid_smiles_final_new.append(smile)
        print(i)

    valid_smiles_final = valid_smiles_final_new
    print(valid_smiles_final)
    print(scores)

    save_object(scores, hparams.sample_file+"/scores{}.dat".format(iteration))
    save_object(valid_smiles_final, hparams.sample_file+"/valid_smiles{}.dat".format(iteration))

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    iteration += 1
    
    print(iteration)
