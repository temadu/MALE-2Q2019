import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches


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
    self.neighbourhoodRadius = max(self.network_dimensions[0], self.network_dimensions[1]) / 2
    # radius decay parameter
    self.radiusDecay = self.totalIterations / np.log(self.neighbourhoodRadius)

  def getBMU(self, t, net, m):
    """
        Find the best matching unit for a given vector, t
        Returns: bmu and bmuIndex is the index of this vector in the SOM
    """
    bmuIndex = np.array([0, 0])
    min_dist = np.iinfo(np.int).max
    
    # calculate the distance between each neuron and the input
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x, y, :].reshape(m, 1)
            sq_dist = np.sum((w - t) ** 2)
            sq_dist = np.sqrt(sq_dist)
            if sq_dist < min_dist:
                min_dist = sq_dist # dist
                bmuIndex = np.array([x, y]) # id
    
    bmu = net[bmuIndex[0], bmuIndex[1], :].reshape(m, 1)
    return (bmu, bmuIndex)

  def decayRadius(self, i):
      return self.neighbourhoodRadius * np.exp(-i / self.radiusDecay)

  def decayLearningRate(self, i):
      return self.learningRate * np.exp(-i / self.totalIterations)

  def calculateInfluence(self, distance, radius):
      return np.exp(-distance / (2* (radius**2)))

  def fit(self, data):
    m = data.shape[0]
    n = data.shape[1]
    print('m: %d, n:%d' % (m,n))
    if self.normalizeData:
      if self.normalizeByColumn:
        colMaxes = data.max(axis=0)
        data = data / colMaxes[np.newaxis, :]
      else:
        data = data / data.max()
    net = np.random.random((self.network_dimensions[0], self.network_dimensions[1], m))

    for i in range(self.totalIterations):
      # select a training example at random
      t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))

      # find its Best Matching Unit
      bmu, bmuIndex = self.getBMU(t, net, m)

      # decay the SOM parameters
      r = self.decayRadius(i)
      l = self.decayLearningRate(i)

      # update weight vector to move closer to input
      # and move its neighbours in 2-D vector space closer
      for x in range(net.shape[0]):
        for y in range(net.shape[1]):
          w = net[x, y, :].reshape(m, 1)
          w_dist = np.sum((np.array([x, y]) - bmuIndex) ** 2)
          w_dist = np.sqrt(w_dist)

          if w_dist <= r:
            # calculate the degree of influence (based on the 2-D distance)
            influence = self.calculateInfluence(w_dist, r)

            # new w = old w + (learning rate * influence * delta)
            # where delta = input vector (t) - old w
            newW = w + (l * influence * (t - w))
            net[x, y, :] = newW.reshape(1, m)

    self.net = net
totalIterations = 10000
raw_data = np.random.randint(0, 255, (3, 100))
ksom = KohonenSOM(normalizeData=False, xDimentions=5, yDimentions=5, totalIterations=totalIterations)
ksom.fit(raw_data)
print(ksom.net)
print(ksom.net.shape)
# fig = plt.figure()

# ax = fig.add_subplot(111, aspect='equal')
# ax.set_xlim((0, ksom.net.shape[0]+1))
# ax.set_ylim((0, ksom.net.shape[1]+1))
# ax.set_title('Self-Organising Map after %d iterations' % totalIterations)

# # plot
# for x in range(1, ksom.net.shape[0] + 1):
#     for y in range(1, ksom.net.shape[1] + 1):
#         ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
#                     facecolor=ksom.net[x-1, y-1, :],
#                     edgecolor='none'))
# plt.show()
