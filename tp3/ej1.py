import matplotlib.pyplot as plt
import numpy as np

X = np.random.rand(100,2)*5
for i in range(len(X)):
  if(7*X[i,0] < 5*X[i,1]):
    plt.scatter(X[i,0],X[i,1], c='r')
  else:
    plt.scatter(X[i,0],X[i,1], c='g')

plt.show()