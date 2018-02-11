import sys

f = open(sys.argv[1])
database_smiles= {}
f2 = open(sys.argv[2])
for line in f2:
    [smiles, idzinc] = line.strip().split(" ")
    database_smiles[idzinc] = smiles

molecules = f.read().split("@<TRIPOS>MOLECULE")
#'''
for molecule in molecules[1:]:
    #print "Debug mol", molecule
    name = molecule.split("@<TRIPOS>BOND")[0].split('\n')[1].strip()

    #print name
    size =  int(molecule.split("@<TRIPOS>BOND")[0].split('\n')[2].strip().split("    ")[0])
    bonds = molecule.split("@<TRIPOS>BOND")[1].strip().split('\n')
    flag = False
    for bond in bonds:
        el = bond.split("   ")
        bt = el[-1].split()[-1]
        if bt == 'ar' or bt == 'am' or bt == 'du' or bt == 'un' or bt == 'nc':

            #count += 1
            flag =True
            break

    if not flag:
                
        #if size in range(27,56) or size == 57:
        if size in range(30, 41):
            smiles = database_smiles[name]
            for bond in bonds:
                tokens = bond.split()
                with open("training/test_graphs/n_"+str(size)+"/"+name+".txt", "a") as fw:
                    fw.write(str(int(tokens[1])-1)+" "+str(int(tokens[2])-1) + " {" + tokens[3] + "}\n")
            
            with open("training/test_graphs/n_"+str(size)+"/smiles.txt", "a") as fw:
                fw.write(smiles+"\n")
            #count += 1
        
        #print size 
    #break
#print count
#'''
