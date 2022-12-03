
import csv, os
cwd = os.getcwd()

row = []
with open(cwd+'/GAN/Generate/flow.txt', 'r') as f:
    for line in f:
        a = line.split()
        row.append(a)

with open(cwd+'/GAN/Generate/flow.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['src', 'dst', 'size', 'times'])
    for l in row:
        writer.writerow(l)


