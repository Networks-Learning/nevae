from __future__ import print_function

import networkx as nx
import argparse
import multiprocessing
from rdkit import Chem
import os
from rdkit.Chem import AllChem
import rdkit
NUM_PROCESSES = 8

def get_arguments():
    parser = argparse.ArgumentParser(description='Convert an rdkit Mol to nx graph, preserving chemical attributes')
    parser.add_argument('smiles', type=str, help='The input file containing SMILES strings representing an input molecules.')
    parser.add_argument('out_file', type=str, help='The output file containing sequence of pickled nx graphs')
    #parser.add_argument('nodes', type=int, help='The number of atoms to be controlled')

    parser.add_argument('--num_processes', type=int, default=NUM_PROCESSES, help='The number of concurrent processes to use when converting.')
    return parser.parse_args()


def convert_weights(weight):
    #weight = weight.strip()
    if weight == rdkit.Chem.rdchem.BondType.SINGLE:
        return 1
    elif weight == rdkit.Chem.rdchem.BondType.DOUBLE:
        return 2
    elif weight == rdkit.Chem.rdchem.BondType.TRIPLE:
        return 3
    else:
        return -1


def mol_to_nx(mol):
    G = nx.Graph()
    atom_list = mol.GetAtoms()
    # Checking if it has only 4 types of atoms C, H, O, N       
    for i in range(len(atom_list)):
        atom = atom_list[i]
        atomic_number = atom.GetAtomicNum()
        if atomic_number == 8 or atomic_number == 1 or atomic_number == 6 or atomic_number == 7:
                continue
        else:
                print("Error")
                return ""
    for i in range(len(atom_list)):
        atom = atom_list[i]
        pos = mol.GetConformer().GetAtomPosition(i)
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic(),
                   coord=(pos.x,pos.y,pos.z)
                   )
    for bond in mol.GetBonds():
        w = convert_weights(bond.GetBondType())
        if w == -1:
            return ''
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   weight=w)
    return G

def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol

def do_all(block, validate=False):
    mol = Chem.MolFromSmiles(block.strip())
    mol1 = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol1)
    #can_smi = Chem.MolToSmiles(mol1)
    G = mol_to_nx_new(mol1)
    return G

def main():
    args = get_arguments()
    f = open(args.smiles)
    molecules = []

    for line in f:
        molecules.append(line.strip())

    p = multiprocessing.Pool(args.num_processes)
    results = p.map(do_all, molecules)
    
    list_G = []
    for smile in molecules:
    #for result in results:
        result = do_all(smile)
        if result != "":
	    list_G.append(result)
    #'''	
    name = 0
    for G in list_G:
        size = len(G.nodes())
        dir_ = args.out_file + "n_"+str(size)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        with open(dir_+"/"+str(name)+".txt", "w") as fw:
            nx.write_edgelist(G, fw, data=True) 
            name += 1

if __name__ == '__main__':
    main()
