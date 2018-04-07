import csv

generesSet = set()
xyz = open('new_tags_generes.csv', "w")

i = 0
f = open('movies.csv', 'r')
reader = csv.reader(f)
for row in reader:
    if i == 0:
        columnTitleRow = row[0]+','+row[2]+"\n"
        xyz.write(columnTitleRow)
    else:
        generes = row[2].split('|')
        for genere in generes:
            rowToWrite = row[0] +','+genere+"\n"
            xyz.write(rowToWrite)
            generesSet.add(genere)
    i += 1
f.close()

i = 0
f = open('tags.csv', 'r')
reader = csv.reader(f)
for row in reader:
    if i!= 0:
        rowToWrite = row[1] +','+row[2]+"\n"
        xyz.write(rowToWrite)
        generesSet.add(row[2])
    i += 1
print(len(generesSet))
f.close()
xyz.close()
