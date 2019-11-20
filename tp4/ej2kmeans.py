import os
import nltk
import re
import numpy as np
import copy
import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class KMeans:
  def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
    self.k = k
    self.tolerance = tolerance
    self.max_iterations = max_iterations
    self.classes = {}
    self.centroids = {}

  def fit(self, data):
    self.centroids = {}

    # Agarro centroides
    for i in range(self.k):
        self.centroids[i] = data[i]

    for i in range(self.max_iterations):
      self.classes = {}
      for i in range(self.k):
        self.classes[i] = []

      # clasifico todinas
      for features in data:
          self.classes[self.pred(features)].append(features)

      previous = dict(self.centroids)

      # Recalculo centroides
      for classification in self.classes:
          self.centroids[classification] = np.average(
              self.classes[classification], axis=0)

      isOptimal = True
      for centroid in self.centroids:
          original_centroid = previous[centroid]
          curr = self.centroids[centroid]
          # if(original_centroid == 0):
          #   isOptimal = False
          if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
            isOptimal = False

      # Corto el algoritmo si los centroides no se mueven
      if isOptimal:
          break

  def pred(self, data):
      distances = [np.linalg.norm(data - self.centroids[centroid])
                    for centroid in self.centroids]
      classification = distances.index(min(distances))
      return classification


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


def calcMinDistClusters(clusters, method=0):  # data es lista de clusters
    min = 999
    coords = [-1, -1]
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            x = clusters[i]
            y = clusters[j]
            dist = np.linalg.norm(np.array(x.centroid) - np.array(y.centroid))
            sumx = 0
            for k, w in enumerate(x.centroid, start=0):
                sumx += (x.centroid[k] - y.centroid[k]) ** 2
            dist = np.sqrt(sumx)
            # print(dist)
            qc = cdist(x.data, y.data)
            if method == 0:
                dist = np.mean(cdist(x.data, y.data)),
            elif method == 1:
                dist = np.amin(cdist(x.data, y.data)),
            elif method == 2:
                dist = np.amax(cdist(x.data, y.data)),
            elif method == 3:
                dist = cdist([x.centroid], [y.centroid]),

            if dist[0] < min and dist[0] != 0:
                min = dist[0]
                if i > j:
                    coords = [i, j]
                else:
                    coords = [j, i]
    return min, coords


def main():
    method = 0
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

    # print(data)
    npdata = np.array(data)
    print(npdata.shape)
    km = KMeans(k=5)
    km.fit(npdata)
    print(km.centroids)
    


main()
