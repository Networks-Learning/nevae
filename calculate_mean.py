import sys

f = open(sys.argv[1])
total = 0.0
count = 0
for line in f:
    total += float(line.strip())
    count +=1

print total/count
