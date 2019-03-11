lineCountSeps = 100
inputFile = "./perflogsimple.txt"

fin = open(inputFile, 'r')

fout = open(inputFile+".csv", 'w')

firstLine=fin.readline()
fout.write(firstLine)

ln = 0

for line in fin:
	if ln % lineCountSeps == 0:
		print("Loading line: ", ln)
		fout.write(line)
	ln+= 1
fin.close()
fout.close()
