import csv

xyz = open('Data/l-20m/new_tags_generes.csv', "a")

i = 0
f = open('tags.csv', 'r')
reader = csv.reader(f)
for row in reader:
    if i == 0:
        continue
    else:
        rowToWrite = row[1] +','+row[2]+"\n"
        xyz.write(rowToWrite)
    i += 1
f.close()
xyz.close()
