import csv
import numpy as np
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def loadDataset(filename, dataset=[]):
  dataset.append([])
  dataset.append([])
  dataset.append([])
  with open(filename, encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    first = True
    for row in reader:
      if first:
        first = False
        continue
      # print(row)
      dataset[0].append(float(row[0]))
      dataset[1].append(float(row[1]))
      dataset[2].append(float(row[2]))


def simpleLinearRegression(x, y):
  x = np.array(x).reshape((-1, 1))
  y = np.array(y)
  model = linear_model.LinearRegression().fit(x, y)
  r_sq = model.score(x, y)
  # y = ax+b
  # print('coefficient of determination:', r_sq)
  print('pendiente a:', model.coef_[0]) 
  print('ordenada al origen b:', model.intercept_)
  print('Y = {0:.4g} X + {1:.4g}'.format(model.coef_[0], model.intercept_))
  return model.coef_, model.intercept_


def multipleLinearRegression(x1, x2, y):
  x = []
  for i in range(len(x1)):
    x.append([x1[i],x2[i]])
  x = np.array(x)
  y = np.array(y)
  model = linear_model.LinearRegression().fit(x, y)
  r_sq = model.score(x, y)
  # y = a1x1 + a2x2 + b 
  # print('coefficient of determination:', r_sq)
  print('pendientes a:', model.coef_) 
  print('ordenada al origen b:', model.intercept_)
  print('Y = {0:.4g} X1 + {1:.4g} X2 + {2:.4g}'.format(
      model.coef_[0], model.coef_[1], model.intercept_))
  return model.coef_, model.intercept_

def plotComparison(rawX,rawY, a, b, nameX, nameY):
  plt.figure()
  abline_values = [a * i + b for i in rawX]
  plt.plot(rawX, rawY, 'o')
  plt.plot(rawX, abline_values, 'b')
  plt.xlabel(nameX)
  plt.ylabel(nameY)
  plt.title(nameX + ' vs. ' + nameY)
  plt.show()

if __name__ == "__main__":
  dataset = []
  loadDataset('data/cervezas.csv', dataset)
  cajas = dataset[0]
  distancia = dataset[1]
  tiempo = dataset[2]
  
  print()
  print("Tiempo vs Cajas")
  a1, b1 = simpleLinearRegression(cajas, tiempo)
  plotComparison(cajas, tiempo, a1, b1, "Cajas", "Tiempo")

  print()
  print("Tiempo vs Distancia")
  a2, b2 = simpleLinearRegression(distancia, tiempo)
  plotComparison(distancia, tiempo, a2, b2, "Distancia", "Tiempo")
  
  print()
  print("Tiempo vs (Cajas,Distancia)")
  multipleLinearRegression(cajas, distancia, tiempo)

  print()
  pass
