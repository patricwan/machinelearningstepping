import csv
import hashlib

def hashstr(S, nr_bins):
    return str(int(hashlib.md5(S.encode('utf-8')).hexdigest(), 16)%(nr_bins-1) + 1)

def writeToFFMFile(csvfile, ffmfile, targetLabel = "Label"):
    print("would writeToFFMFile")
    print(csvfile)
    with open(ffmfile, 'w') as f:
        for row in csv.DictReader(open(csvfile)):
            #print(row)
            # row is dict
            row_to_write = [row[targetLabel], ]
            field = 0
            for feat in row.keys():
                if feat == targetLabel:
                    continue
                items = str(row[feat]).split(" ")
                for item in items:
                    row_to_write.append(":".join([str(field), hashstr(str(field)+'='+item, int(1e+6)), '1']))
                field += 1
            row_to_write = " ".join(row_to_write)
            f.write(row_to_write + '\n')
