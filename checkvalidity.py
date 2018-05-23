import sys
import rdkit
from rdkit import Chem
import networkx as nx
import numpy as np
from collections import defaultdict
import glob 
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, Lipinski
from rdkit.Chem import Descriptors
#import sascorer

def getValenceDict(filename):
    f = open(filename)
    jobj = json.load(f.read())
    return jobj

def getAtom(valency):
    atom = "Un"
    if valency == 4:
        atom = 'C'
    if valency == 3:
        atom = 'N'
    if valency == 2:
        atom = 'O'
    if valency == 1:
        atom = 'H'
    if valency == 6:
        atom = 'S' 
    return atom

def guess_correct_molecules(readfile, writefile, n, multi):
    
    f = open(readfile)
    try:
        G=nx.read_edgelist(f, nodetype=int)
    except:
        f = open(readfile)
        lines = f.read()
        linesnew = lines.replace("{", "{\'weight\':").split("\n")
        G = nx.parse_edgelist(linesnew, nodetype=int)
        #return False
    #'''
    try:
     if not nx.is_connected(G):
           print "Not connected"
           return False
    except:
        print "Null graph"
        return False
    #'''
    nodes = len(G.nodes())
    count = 1 
    index = defaultdict(int)
    for i in range(n):
        if i not in G.nodes():
            G.add_node(i)
        else:
            index[i] = count
            count += 1

    e = len(G.edges())
    deg = [] 
    adj = np.array(nx.adjacency_matrix(G).todense())
    for i in G.nodes():
        deg.append(np.sum(adj[i]))
    
    deg = np.array(deg)
    #print "debug", deg
    maxdeg = deg.max()
    if maxdeg >= 5:
        return False
    #if maxdeg >= 7 or maxdeg == 5 :
    #    return False
    #print degarray
    CC = 0 
    HC = 0
    NC = 0
    OC = 0
    SC = 0

    fw = open(writefile, "w")
    
    fw.write("@<TRIPOS>MOLECULE\n")
    fw.write("Dummy Atom\n")
    fw.write(str(nodes)+" "+str(e)+"\n\n")
    
    fw.write("@<TRIPOS>ATOM\n")
    
    atom_count = 0
    #print "DE", len(degarray), degarray
    for i in range(n):
        #print "Debug", i , deg[i]
        if deg[i] == 0:
            continue
        atom = getAtom(deg[i])
        atom_count += 1
        if atom == 'C':
            CC+=1
            fw.write(str(atom_count)+"\t"+atom+str(CC)+"\t0\t0\t0\tC\n")
        elif atom == 'N':
            NC+=1
            fw.write(str(atom_count)+"\t"+atom+str(NC)+"\t0\t0\t0\tN\n")
        elif atom == 'H':
            HC+=1
            fw.write(str(atom_count)+"\t"+atom+str(HC)+"\t0\t0\t0\tH\n")
        elif atom == 'O':
            OC+=1
            fw.write(str(atom_count)+"\t"+atom+str(OC)+"\t0\t0\t0\tO\n")
        
        elif atom=='S':
            SC+=1
            fw.write(str(atom_count)+"\t"+atom+str(SC)+"\t0\t0\t0\tS\n")
        else:
            fw.write(str(atom_count)+"\t"+atom+"\t0\t0\t0\tUN\n")
    fw.write("\n@<TRIPOS>BOND\n")
    edge_count = 0
    for (u,v) in G.edges(): #list(G.edges_iter(data='weight', default=1)):
        edge_count += 1
        w = G.get_edge_data(u,v)['weight']
        fw.write(str(edge_count) + "\t" + str(index[u]) + "\t" + str(index[v]) + "\t" + str(w) + "\n")
    return True    

def drawchem(mols):
    count = 0
    mollist = []
    for (mol, filename) in mols:
        tokens = filename.split('/')[-1].replace(".txt", "")
        #print tokens
        mol = Chem.RemoveHs(mol)
        AllChem.Compute2DCoords(mol)
        #mollist.append(mol)
        
        img = Draw.MolsToGridImage([mol], molsPerRow = 1, subImgSize=(300, 300), useSVG=True)
        with open(sys.argv[5]+tokens+".svg", "w") as text_file:
                #text_file.write(img)
                Draw.MolToFile(mol,sys.argv[5]+tokens+'.svg', size=(300, 200))
        count += 1 

def calculate_property(m):
    SA_score = -sascorer.calculateScore(m)
    MW = Descriptors.MolWt(m)
    RB = Lipinski.NumRotatableBonds(m)
    logp = Descriptors.MolLogP(m)
    return (SA_score, MW, RB, logp)

if __name__=="__main__":
    
    readfileDir = sys.argv[1] + '*'
    writefile = sys.argv[2]

    total = 0.0
    valid = 0.0
    moltotal = 0
    smiles = []
    mols = []
    invalid = []

    for readfile in sorted(glob.glob(readfileDir)):
        moltotal +=1
        #print readfile
        if guess_correct_molecules(readfile, writefile, int(sys.argv[3]), int(sys.argv[4])):
            total += 1
            m1 = Chem.MolFromMol2File(sys.argv[2])
            if m1 != None:
                s = Chem.MolToSmiles(m1)
                m2 = Chem.AddHs(m1)
                smiles.append(s)
                mols.append((m2, readfile))
                valid += 1
        else:
                invalid.append(readfile)

    #print smiles, len(smiles)
    with open('smiles.smi', 'w') as f:
        for smile in smiles:
            f.write(smile+"\n")
    print "Valid:", valid, "Total:", total, "moltotal:",moltotal, "Perc:", valid/moltotal
    #drawchem(mols)
