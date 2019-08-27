import csv

import numpy

rows = [];
with open('./data/PreferenciasBritanicos.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        rows.append(row)
csvFile.close()
# print(rows)

pE = 7/13
pI = 6/13

probs = [0,0,0,0,0]
probsE = [0,0,0,0,0]
probsI = [0,0,0,0,0]
amount = len(rows) - 1
for i in range(1, len(rows)):
    for j in range(0, len(rows[0])-1):
        if rows[i][j] == '1':
            probs[j] += 1/amount
            if rows[i][5] == "E":
                probsE[j] += 1/7
            elif rows[i][5] == "I":
                probsI[j] += 1/6

probToTest = [1, 0, 1, 1, 0]
# print(probsE)
# print(probsI)

for i in range(0, 5):
    if probToTest[i] == 0:
        probsE[i] = 1 - probsE[i]
        probsI[i] = 1 - probsI[i]
# print(probs)
# print(probsE)
# print(probsI)

cacaE = numpy.prod(probsE) * pE
cacaI = numpy.prod(probsI) * pI

# print(cacaE)
# print(cacaI)
if max(cacaE, cacaI) == cacaE:
    print("ENGLISH")
else:
    print("IRELISH")

