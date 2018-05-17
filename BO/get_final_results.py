import sys
import pickle
import gzip
from collections import defaultdict

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import sascorer
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

# We compute the average statistics for the grammar autoencoder

import numpy as np

n_simulations = 1
iteration = int(sys.argv[2])
results_grammar = np.zeros((n_simulations, 3))
mols = defaultdict(list)

for j in range(1, n_simulations + 1):
    best_value = 1e10
    n_valid = 0
    max_value = 0
    #-1e10
    for i in range(iteration):
        #smiles = load_object('results{}/valid_smiles{}.dat'.format(j, i))
        #scores = load_object('results{}/scores{}.dat'.format(j, i))
       
        smiles = load_object(sys.argv[1]+'/valid_smiles{}.dat'.format(i))
        scores = load_object(sys.argv[1]+'/scores{}.dat'.format(i))
        
        #print("Scores", scores, np.argsort(scores))
        scoressorted = np.argsort(scores)
        print("Best", -scores[scoressorted[0]], smiles[scoressorted[0]])
        mols[i].append(MolFromSmiles(smiles[scoressorted[0]]))
        print("2nd Best", -scores[scoressorted[1]], smiles[scoressorted[1]])
        mols[i].append( MolFromSmiles(smiles[scoressorted[1]]))
        print("3rd Best", -scores[scoressorted[2]], smiles[scoressorted[2]])
        mols[i].append( MolFromSmiles(smiles[scoressorted[2]]))
        print("Worst", -scores[scoressorted[-1]], smiles[scoressorted[-1]])
        mols[i].append( MolFromSmiles(smiles[scoressorted[-1]]))
        
        #print("worst", )
        n_valid += len([ x for x in smiles if x is not None ])
        
        if min(scores) < best_value:
            best_value = min(scores)
        #if 2nd_best > best_value and  min(score)<
        
        if max(scores) > max_value:
            max_value = max(scores)

    import numpy as np

    sum_values = 0
    count_values = 0
    less_zero = 0
    for i in range(iteration):
        scores = np.array(load_object(sys.argv[1]+'/scores{}.dat'.format(i)))
        sum_values += np.sum(scores[  scores < max_value ])
        count_values += len(scores[  scores < max_value ])
        less_zero += len(scores[scores<0])
    print("Count good", less_zero, len(scores) * iteration)
    print("Min value", -max_value )
    # fraction of valid smiles
    results_grammar[ j - 1, 0 ] = 1.0 * n_valid / (iteration * 50)
    # Best value
    results_grammar[ j - 1, 1 ] = best_value
    # Average value = 
    results_grammar[ j - 1, 2 ] = 1.0 * sum_values / count_values

print("Results VAE (fraction valid, best, average)): Stat per simulation")
print("Mean:", np.mean(results_grammar, 0)[ 0 ], -np.mean(results_grammar, 0)[ 1 ], -np.mean(results_grammar, 0)[ 2 ])
print("Std:", np.std(results_grammar, 0) / np.sqrt(iteration))
print("First:", -np.min(results_grammar[ : , 1 ]))
print("Debug", results_grammar[:,1])
best_score = np.min(results_grammar[ : , 1 ])
results_grammar[ results_grammar[ : , 1 ] == best_score , 1 ] = 1e10
print("Second:", -np.min(results_grammar[ : , 1 ]))
second_best_score = np.min(results_grammar[ : , 1 ])
results_grammar[ results_grammar[ : , 1 ] == second_best_score, 1 ] = 1e10
print("Third:", -np.min(results_grammar[ : , 1 ]))
third_best_score = np.min(results_grammar[ : , 1 ])

#from rdkit.Chem import MolFromSmiles, MolToSmiles
#from rdkit.Chem import Draw
#from rdkit.Chem import Descriptors
for i in range(iteration):
    #img = Draw.MolsToGridImage([mols[i][0]], molsPerRow = len([mols[i][0]]), subImgSize=(300, 300), useSVG=True)
    with open(sys.argv[1]+"/molecule_images/best_VG_molecule_"+str(i)+"best.svg", "w") as text_file:
        #text_file.write(img)
        Draw.MolToFile(mols[i][0],sys.argv[1]+"/molecule_images/best_VG_molecule_"+str(i)+"best.svg", size=(300,200))
    print "FIsrt ", sascorer.calculateScore(mols[i][0]) , Descriptors.MolLogP(mols[i][0])   
    #img = Draw.MolsToGridImage([mols[i][1]], molsPerRow = len([mols[i][0]]), subImgSize=(300, 300), useSVG=True)
    with open(sys.argv[1]+"/molecule_images/best_VG_molecule_"+str(i)+"2ndbest.svg", "w") as text_file:
        #text_file.write(img)
        Draw.MolToFile(mols[i][1],sys.argv[1]+"/molecule_images/best_VG_molecule_"+str(i)+"2ndbest.svg", size=(300,200))
    print "2nd ", sascorer.calculateScore(mols[i][1]), Descriptors.MolLogP(mols[i][1]) 

    #img = Draw.MolsToGridImage([mols[i][2]], molsPerRow = len([mols[i][0]]), subImgSize=(300, 300), useSVG=True)
    with open(sys.argv[1]+"/molecule_images/best_VG_molecule_"+str(i)+"3rdbest.svg", "w") as text_file:
        #text_file.write(img)
        Draw.MolToFile(mols[i][2], sys.argv[1]+"/molecule_images/best_VG_molecule_"+str(i)+"3rdbest.svg", size=(300,200))
    print "3rd ", sascorer.calculateScore(mols[i][2]), Descriptors.MolLogP(mols[i][2]) 



    #img = Draw.MolsToGridImage([mols[i][3]], molsPerRow = len([mols[i][0]]), subImgSize=(300, 300), useSVG=True)
    with open(sys.argv[1]+"/molecule_images/best_VG_molecule_"+str(i)+"worst.svg", "w") as text_file:
        Draw.MolToFile(mols[i][3],sys.argv[1]+"/molecule_images/best_VG_molecule_"+str(i)+"worst.svg", size=(300,200))
        #text_file.write(img)

    #print "worst ", sascorer.calculateScore(mols[i][3]), Descriptors.MolLogP(mols[i][3]) 

