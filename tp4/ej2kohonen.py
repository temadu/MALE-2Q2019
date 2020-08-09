import os
import nltk
import re
import numpy as np
import copy
import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class KohonenSOM:
  def __init__(self, xDimentions=5, yDimentions=5,
               learningRate=0.01, totalIterations=10000,
               normalizeData=True, normalizeByColumn=False):
    self.network_dimensions = np.array([yDimentions, xDimentions])
    self.learningRate = learningRate
    self.totalIterations = totalIterations
    self.normalizeData = normalizeData
    self.normalizeByColumn = normalizeByColumn

    # initial neighbourhood radius
    self.neighbourhoodRadius = max(
        self.network_dimensions[0], self.network_dimensions[1]) / 2
    # radius decay parameter
    self.radiusDecay = self.totalIterations / np.log(self.neighbourhoodRadius)

  def getBMU(self, t, m):
    """
        Find the best matching unit for a given vector, t
        Returns: bmu and bmuIndex is the index of this vector in the SOM
    """
    bmuIndex = np.array([0, 0])
    min_dist = np.iinfo(np.int).max

    # calculate the distance between each neuron and the input
    for x in range(self.net.shape[0]):
        for y in range(self.net.shape[1]):
            w = self.net[x, y, :].reshape(m, 1)
            sq_dist = np.sum((w - t) ** 2)
            sq_dist = np.sqrt(sq_dist)
            if sq_dist < min_dist:
                min_dist = sq_dist  # dist
                bmuIndex = np.array([x, y])  # id

    bmu = self.net[bmuIndex[0], bmuIndex[1], :].reshape(m, 1)
    return (bmu, bmuIndex)

  def decayRadius(self, i):
      return self.neighbourhoodRadius * np.exp(-i / self.radiusDecay)

  def decayLearningRate(self, i):
      return self.learningRate * np.exp(-i / self.totalIterations)

  def calculateInfluence(self, distance, radius):
      return np.exp(-distance / (2 * (radius**2)))

  def fit(self, data):
    m = data.shape[0]
    n = data.shape[1]
    print('m: %d, n:%d' % (m, n))
    if self.normalizeData:
      if self.normalizeByColumn:
        colMaxes = data.max(axis=0)
        data = data / colMaxes[np.newaxis, :]
      else:
        data = data / data.max()
    self.net = np.random.random(
        (self.network_dimensions[0], self.network_dimensions[1], m))

    for i in range(self.totalIterations):
      # select a training example at random
      t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))

      # find its Best Matching Unit
      bmu, bmuIndex = self.getBMU(t, m)

      # decay the SOM parameters
      r = self.decayRadius(i)
      l = self.decayLearningRate(i)

      # update weight vector to move closer to input
      # and move its neighbours in 2-D vector space closer
      for x in range(self.net.shape[0]):
        for y in range(self.net.shape[1]):
          w = self.net[x, y, :].reshape(m, 1)
          w_dist = np.sum((np.array([x, y]) - bmuIndex) ** 2)
          w_dist = np.sqrt(w_dist)

          if w_dist <= r:
            # calculate the degree of influence (based on the 2-D distance)
            influence = self.calculateInfluence(w_dist, r)

            # new w = old w + (learning rate * influence * delta)
            # where delta = input vector (t) - old w
            newW = w + (l * influence * (t - w))
            self.net[x, y, :] = newW.reshape(1, m)


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
    npdatat = npdata.transpose()
    print(npdatat.shape)
    ksom = KohonenSOM(xDimentions=5,
                      yDimentions=5, totalIterations=1000)
    ksom.fit(npdatat)
    print(ksom.net)
    print(ksom.net.shape)
    


main()
