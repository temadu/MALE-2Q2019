import matplotlib.pyplot as plt
import numpy as np

X = np.random.rand(10, 2) * 5
for i in range(len(X)):
    if 7 * X[i, 0] < 5 * X[i, 1]:
        plt.scatter(X[i, 0], X[i, 1], c='r')
    else:
        plt.scatter(X[i, 0], X[i, 1], c='g')

x = np.linspace(0, 3.5, 1000)
# plt.plot(x, 7 / 5 * x)


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

    def printWeights(self):
        print(str(self.weights[1]) + " * x + " + str(self.weights[2]) + " * y + " + str(self.weights[0]) + " = 0")
        return self.weights


def distToLine(x0, y0, a, b, c):
    return np.abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)


training_inputs = X

labels = np.array([0 if 7 * x[0] < 5 * x[1] else 1 for x in X])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

minDist1 = 9999
minDist0 = 9999
minPoint1 = []
minPoint0 = []

w = perceptron.printWeights()
# plt.show()

for row, label in zip(training_inputs, labels):
    dist = distToLine(row[0], row[1], w[1], w[2], w[0])
    if label == 0:
        if dist < minDist0:
            minDist0 = dist
            minPoint0 = row
    else:
        if dist < minDist1:
            minDist1 = dist
            minPoint1 = row

inputs = np.array([3, 2])
print(perceptron.predict(inputs))

inputs = np.array([3, 1])
print(perceptron.predict(inputs))


xRoot = -w[0] / w[1]
yRoot = -w[0] / w[2]
if xRoot == 0:
  m = -w[1]/w[2]
else:
  m = yRoot / -xRoot

b0 = minPoint0[1] - m*minPoint0[0]
b1 = minPoint1[1] - m*minPoint1[0]
x = np.linspace(-yRoot/m if -yRoot/m > 0 else 0, (5 - yRoot)/m, 1000)
plt.plot(x, m * x + yRoot)
x = np.linspace(-b0/m if -b0/m > 0 else 0, (5 - b0)/m, 1000)
plt.plot(x, m * x + b0)
x = np.linspace(-b1/m if -b1/m > 0 else 0, (5 - b1)/m, 1000)
plt.plot(x, m * x + b1)
optimal = (b1 + b0)/2.0
x = np.linspace(-optimal/m if -optimal/m > 0 else 0, (5 - optimal)/m, 1000)
plt.plot(x, m * x + optimal, '--')
plt.show()

print("{}*x+{}".format(m, yRoot))



