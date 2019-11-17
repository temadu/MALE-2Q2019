import os
import nltk
import re
import numpy as np


# Cantidad de Conjunciones subordinantes: En las oraciones con conjunciones
# subordinantes, existe una oraci´on principal y una oraci´on secundaria que es introducida por la conjunci´on subordinante y que depende de la principal.
# • Conjunciones subordinantes causales: porque, pues, ya que, puesto que, a causa
# de, debido a.
# • Conjunciones subordinantes consecutivas o ilativas: luego, conque, as´ı que.
# • Conjunciones subordinantes condicionales: si.
# • Conjunciones subordinantes finales: para que, a fin de que.
# • Conjunciones subordinantes comparativas: como, que.
# • Conjunciones subordinantes concesivas: aunque, aun cuando, si bien.
# • Conjunciones subordinantes completivas: que, si.
# Cantidad de conjunciones coordinantes: Unen palabras u oraciones que tengan
# la misma jerarqu´ıa.
# • ni, y, o, o bien, pero aunque, no obstante, sin embargo, sino, por el contrario.
# Frecuencia relativa de art´ıculos determinados: La, el, los, las.
# Frecuencia relativa de art´ıculos indeterminados: un, una unos, unas.
# Cantidad de adverbios que terminen en mente.

def getTextVariables(text):
    # words per sentence, common 5, uniquewords, articulos def, articulos indef, mente, csconsec, cscond, csfina, cscompar, csconces, cscomplet, ccoordin
    attributes = []
    text = text.lower()
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    wordsFull = tokenizer.tokenize(text)
    freq = nltk.FreqDist(wordsFull)
    # print(freq.most_common())
    # words per sentence
    sents = nltk.sent_tokenize(text, 'spanish')
    avgWpS = 1 / len(sents)
    attributes.append(avgWpS)

    # freq 5 common words
    total = 0
    for x in freq.most_common(5):
        total += x[1]
    attributes.append(total/freq.N())
    # unique words
    attributes.append(freq.B()/freq.N())
    # articulos
    arts = 0
    arts += freq['la']
    arts += freq['el']
    arts += freq['las']
    arts += freq['los']
    attributes.append(arts/freq.N())
    # articulos undef
    arts = 0
    arts += freq['un']
    arts += freq['una']
    arts += freq['unos']
    arts += freq['unas']
    attributes.append(arts / freq.N())

    # mente
    mente = [w for w in wordsFull if re.search('.mente$', w)]
    attributes.append(len(mente) / freq.N())

# • Conjunciones subordinantes causales:                    porque, pues, ya que, puesto que, a causa de, debido a.
    count = 0
    for m in re.finditer("(ya que|puesto que|a causa de|debido a)", text):
        count += 1

    arr = [m for m in wordsFull if re.search("^(porque|pues)$", m)]
    attributes.append((len(arr) + count) / freq.N())
# • Conjunciones subordinantes consecutivas o ilativas:     luego, conque, as´ı que.
    count = 0
    for m in re.finditer("(así que)", text):
        count += 1

    arr = [m for m in wordsFull if re.search("^(luego|conque)$", m)]
    attributes.append((len(arr) + count) / freq.N())

# • Conjunciones subordinantes condicionales:               si.
    count = 0
    arr = [m for m in wordsFull if re.search("^(si)$", m)]
    attributes.append((len(arr) + count) / freq.N())

# • Conjunciones subordinantes finales:                     para que, a fin de que.
    count = 0
    for m in re.finditer("(para que|a fin de que)", text):
        count += 1
    attributes.append((len(arr) + count) / freq.N())


# • Conjunciones subordinantes comparativas:                como, que.
    count = 0
    arr = [m for m in wordsFull if re.search("^(como|que)$", m)]
    attributes.append((len(arr) + count) / freq.N())

# • Conjunciones subordinantes concesivas:                  aunque, aun cuando, si bien.
    count = 0
    for m in re.finditer("(aun cuando|si bien)", text):
        count += 1

    arr = [m for m in wordsFull if re.search("^(aunque)$", m)]
    attributes.append((len(arr) + count) / freq.N())

# • Conjunciones subordinantes completivas:                 que, si
    count = 0
    arr = [m for m in wordsFull if re.search("^(que|si)$", m)]
    attributes.append((len(arr) + count) / freq.N())

# • Conjunciones coordinantes:  ni, y, o, o bien, pero aunque, no obstante, sin embargo, sino, por el contrario.
    count = 0
    for m in re.finditer("(o bien|pero aunque|no obstante| sin embargo| por el contrario)", text):
        count += 1

    arr = [m for m in wordsFull if re.search("^(ni|y|o|sino)$", m)]
    attributes.append((len(arr) + count) / freq.N())

    return attributes


def calcProximityMatrix(data): #data es lista de clusters
    mat = np.zeros([len(data), len(data)], dtype=float)
    for i, x in enumerate(data, start=0):
        for j, y in enumerate(data, start=0):
            mat[i][j] = np.linalg.norm(np.array(x.centroid) - np.array(y.centroid))
    return mat


def findMinCoords(matrix):
    min = 1
    minCoords = [-1, -1]
    for i, x in enumerate(matrix):
        for j, y in enumerate(matrix):
            if matrix[i][j] < min and matrix[i][j] != 0:
                min = matrix[i][j]
                minCoords = [i, j]

    return minCoords, min


def calcClusterCentroid(elems):
    return np.average(elems, axis=0)

data = []
staticData = []
for filename in os.listdir('./data/'):
    if filename.endswith(".txt"):
        with open(os.path.join('./data/', filename), "r", encoding='ansi', errors='replace') as textFile:
            print(filename)
            text = textFile.read()
            variabs = getTextVariables(text)
            data.append(variabs)
            staticData.append(variabs)

print(data)
historyDists = []
historyClusters = []
historyCoords = []
dists = calcProximityMatrix(data)
clusters = list(range(0, len(data)))
historyClusters.append(clusters)
mc, m = findMinCoords(dists)
historyDists.append(m)
historyCoords.append(mc)
print(mc) #20, 22 la primera

for x in clusters:
    if x == clusters[mc[0]] or x == clusters[mc[1]]:
        clusters[x] = clusters[mc[0]]

auxClusters = set(clusters)
data = []
realIndices = []
for x in auxClusters:
    elems = []
    for i, y in enumerate(clusters):
        if x == y:
            elems.append(np.array(staticData[i]))
    centroid = calcClusterCentroid(np.asarray(elems))
    data.append(centroid)
    realIndices.append(x)





# [1, 2, 1, 4]
# [1, 2, 4]

class Cluster:
    def __init__(self, index, members, membersData):
        self.index = index
        self.members = members
        self.data = membersData
        self.centroid = np.average(membersData, axis=0)


    # def getDistance(self, otherCluster):




























