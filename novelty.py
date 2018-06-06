import sys

if __name__ == "__main__":
    f = open(sys.argv[1])
    database_smiles = []
    molcount = 0
    c_count = 0
    for line in f:
	molcount += 1
        database_smiles.append(line.strip())
    f.close()
    count = 0.0
    f = open(sys.argv[2])
    
    for line in f:
        line = line.strip()
        #print line
        if line in database_smiles:
            count +=1
    print "common", count
    print "novel", (1-count/molcount)
