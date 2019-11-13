import csv
import math
import random
import operator
import pandas as pd
import cv2 as cv
import numpy as np
from sklearn import linear_model, metrics, neighbors, cluster
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def calculateConfusionMatrix(testValues, predictions, posibleValues, printit=True):
    allClasses = {}
    for i, c in enumerate(posibleValues):
        allClasses[c] = i
    confusionMatrix = [[0 for x in range(len(posibleValues))]
                       for y in range(len(posibleValues))]
    fp = 0
    fn = 0

    for i in range(len(testValues)):
        if(predictions[i] == 1 and testValues[i] != 1):
            fp += 1
        if(predictions[i] != 1 and testValues[i] == 1):
            fn += 1
        confusionMatrix[allClasses[testValues[i]]
                        ][allClasses[predictions[i]]] += 1
    if(printit):
        print("CONFUSION MATRIX")
        print(pd.DataFrame(confusionMatrix,
              columns=posibleValues, index=posibleValues))
        print()

    return calculateStatistics(posibleValues, confusionMatrix, printit)


def calculateStatistics(posibleValues, confusionMatrix, printit=True):
    allClasses = {}
    for i, c in enumerate(posibleValues):
        allClasses[c] = i

    evaluationMeasurementsRowNames = list(allClasses.keys()).copy()
    evaluationMeasurementsRowNames.append("TOTAL")
    evaluationMeasurementsTitles = [
        "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1Score"]
    evaluationMeasurements = [[0 for x in range(len(
        evaluationMeasurementsTitles))] for y in range(len(evaluationMeasurementsRowNames))]
    for i in range(len(evaluationMeasurementsRowNames)-1):
        for j in range(len(allClasses)):
            for k in range(len(allClasses)):
                if i == j and i == k:
                    evaluationMeasurements[i][0] = confusionMatrix[j][k]
                    evaluationMeasurements[len(
                        allClasses)][0] += confusionMatrix[j][k]
                elif i == k:
                    evaluationMeasurements[i][3] += confusionMatrix[j][k]
                    evaluationMeasurements[len(
                        allClasses)][3] += confusionMatrix[j][k]
                elif i == j:
                    evaluationMeasurements[i][2] += confusionMatrix[j][k]
                    evaluationMeasurements[len(
                        allClasses)][2] += confusionMatrix[j][k]
                else:
                    evaluationMeasurements[i][1] += confusionMatrix[j][k]
                    evaluationMeasurements[len(
                        allClasses)][1] += confusionMatrix[j][k]

        try:  # Accuracy tp+tn/all
            evaluationMeasurements[i][4] = (evaluationMeasurements[i][0] + evaluationMeasurements[i][1]) / \
                (evaluationMeasurements[i][0]+evaluationMeasurements[i][1] +
                 evaluationMeasurements[i][2]+evaluationMeasurements[i][3])
        except:
            evaluationMeasurements[i][4] = -1

        try:  # Precision tp/(fp+tp)
            evaluationMeasurements[i][5] = (
                evaluationMeasurements[i][0])/(evaluationMeasurements[i][0]+evaluationMeasurements[i][2])
        except:
            evaluationMeasurements[i][5] = -1

        try:  # Accuracy tp/(fp+tp)
            evaluationMeasurements[i][6] = (
                evaluationMeasurements[i][0])/(evaluationMeasurements[i][0]+evaluationMeasurements[i][3])
        except:
            evaluationMeasurements[i][6] = -1

        try:  # Accuracy tp/(fp+tp)
            evaluationMeasurements[i][7] = (
                evaluationMeasurements[i][1])/(evaluationMeasurements[i][1]+evaluationMeasurements[i][2])
        except:
            evaluationMeasurements[i][7] = -1

        try:
            evaluationMeasurements[i][8] = (2*evaluationMeasurements[i][0])/(
                2*evaluationMeasurements[i][0]+evaluationMeasurements[i][2]+evaluationMeasurements[i][3])
        except:
            evaluationMeasurements[i][8] = -1
    if(printit):
        print("EVALUATION MEASUREMENTS")
        print(pd.DataFrame(evaluationMeasurements,
                           columns=evaluationMeasurementsTitles, index=evaluationMeasurementsRowNames))
        print()

    return evaluationMeasurements[0][4]


def loadDataset(filename, dataset=[], normalize=True):
    dataset.append([])
    dataset.append([])
    dataset.append([])
    dataset.append([])
    dataset.append([])
    with open(filename, encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        first = True
        removed2 = 0
        removed4 = 0
        maxAge = 0
        maxBadDays = 0
        maxCholesterol = 0
        for row in reader:
            if first:
                first = False
                continue
            if(row[2] == ""):
                removed2 += 1
                continue
            if(row[4] == ""):
                removed4 += 1
                continue
            tupl = []
            if(float(row[0]) > maxAge):
                maxAge = float(row[0])
            if(float(row[1]) > maxBadDays):
                maxBadDays = float(row[1])
            if(float(row[2]) > maxCholesterol):
                maxCholesterol = float(row[2])

            dataset[0].append(float(row[0]))
            dataset[1].append(float(row[1]))
            dataset[2].append(float(row[2]))
            dataset[3].append(int(row[3]))
            dataset[4].append(int(row[4]))

        # print("max age: " + str(maxAge))
        # print("max days: " + str(maxBadDays))
        # print("max cholesterol: " + str(maxCholesterol))
        if(normalize):
          for i in range(len(dataset[0])):
              dataset[0][i] = dataset[0][i]/maxAge
              dataset[1][i] = dataset[1][i]/maxBadDays
              dataset[2][i] = dataset[2][i]/maxCholesterol
        return dataset


def shuffleDataset(dataset=[]):
    datasetT = np.array(dataset).transpose().tolist()
    random.shuffle(datasetT)
    dataset = np.array(datasetT).transpose().tolist()
    return dataset


def divideDataset(dataset=[], split=0.5, trainingSet=[], testSet=[]):
    datasetT = np.array(dataset).transpose().tolist()
    trainingSet.extend(datasetT[:math.floor(len(datasetT)*split)])
    testSet.extend(datasetT[math.floor(len(datasetT)*split):])
    trainingSet = np.array(trainingSet).transpose().tolist()
    testSet = np.array(testSet).transpose().tolist()
    return trainingSet, testSet


def multipleLogisticRegression(xs, y):
    xs = np.array(xs)
    y = np.array(y)
    model = linear_model.LogisticRegression().fit(xs, y)
    r_sq = model.score(xs, y)
    # y = a1x1 + a2x2 + b
    print('coefficient of determination:', r_sq)
    print('pendientes a:', model.coef_)
    # print('ordenada al origen b:', model.intercept_)
    # print('Y = {0:.4g} X1 + {1:.4g} X2 + {2:.4g}'.format(
    #     model.coef_[0], model.coef_[1], model.intercept_))
    return model.coef_, model.intercept_


def logisticRegressionInclusive(trainSet=[], testSet=[]):
    xs = []
    y = []
    for i in range(len(trainSet[0])):
        xs.append([trainSet[0][i], trainSet[1][i], trainSet[2][i]])
        y.append(trainSet[4][i])
    xs = np.array(xs)
    y = np.array(y)
    model = linear_model.LogisticRegression(
        solver='lbfgs', random_state=0).fit(xs, y)
    r_sq = model.score(xs, y)

    print()
    print('coefficient of determination:', r_sq)
    print('pendientes a:', model.coef_)
    print('ordenada al origen b:', model.intercept_)

    x2s = []
    y2 = []
    for i in range(len(testSet[0])):
        x2s.append([testSet[0][i], testSet[1][i], testSet[2][i]])
        y2.append(testSet[4][i])
    x2s = np.array(x2s)
    y2 = np.array(y2)
    predictions = model.predict(x2s)
    # y = a1x1 + a2x2 + b
    # print('Y = {0:.4g} X1 + {1:.4g} X2 + {2:.4g}'.format(
    #     model.coef_[0], model.coef_[1], model.intercept_))
    # return model.coef_, model.intercept_

    print()
    # cm = metrics.confusion_matrix(y2,predictions)
    # print("Confusion Matrix : \n", cm)
    # print("Accuracy : ", metrics.accuracy_score(y2, predictions))

    calculateConfusionMatrix(y2, predictions,
                             [0, 1], printit=True)

    p = model.predict_proba(np.array([[60, 2, 199]]))
    # print(x2s.shape)
    # print(model.classes_)
    # print(np.array([[10,1,190]]).shape)
    # p = model.predict(np.array([[10, 1, 190]]))
    print(p)
    print()


def logisticRegressionExclusive(trainSet=[], testSet=[]):
    xs = []
    y = []
    for i in range(len(trainSet[0])):
        xs.append([trainSet[0][i], trainSet[1][i],
                  trainSet[2][i], trainSet[3][i]])
        y.append(trainSet[4][i])
    xs = np.array(xs)
    y = np.array(y)
    model = linear_model.LogisticRegression(
        solver='lbfgs', random_state=0).fit(xs, y)
    r_sq = model.score(xs, y)

    print()
    print('coefficient of determination:', r_sq)
    print('pendientes a:', model.coef_)
    print('ordenada al origen b:', model.intercept_)

    x2s = []
    y2 = []
    for i in range(len(testSet[0])):
        x2s.append([testSet[0][i], testSet[1][i],
                   testSet[2][i], testSet[3][i]])
        y2.append(testSet[4][i])
    x2s = np.array(x2s)
    y2 = np.array(y2)
    predictions = model.predict(x2s)
    # y = a1x1 + a2x2 + b
    # print('Y = {0:.4g} X1 + {1:.4g} X2 + {2:.4g}'.format(
    #     model.coef_[0], model.coef_[1], model.intercept_))
    # return model.coef_, model.intercept_

    print()
    # cm = metrics.confusion_matrix(y2,predictions)
    # print("Confusion Matrix : \n", cm)
    # print("Accuracy : ", metrics.accuracy_score(y2, predictions))

    calculateConfusionMatrix(y2, predictions,
                             [0, 1], printit=True)

    p = model.predict_proba(np.array([[60, 2, 199, 0]]))
    print("Man: " + repr(p))
    p = model.predict_proba(np.array([[60, 2, 199, 1]]))
    print("Woman: " + repr(p))
    print()


def KNNInclusive(trainSet=[], testSet=[]):
    xs = []
    y = []
    for i in range(len(trainSet[0])):
        xs.append([trainSet[0][i], trainSet[1][i], trainSet[2][i]])
        y.append(trainSet[4][i])
    xs = np.array(xs)
    y = np.array(y)
    model = neighbors.KNeighborsClassifier(n_neighbors=5).fit(xs, y)

    x2s = []
    y2 = []
    for i in range(len(testSet[0])):
        x2s.append([testSet[0][i], testSet[1][i], testSet[2][i]])
        y2.append(testSet[4][i])
    x2s = np.array(x2s)
    y2 = np.array(y2)
    predictions = model.predict(x2s)
    # y = a1x1 + a2x2 + b
    # print('Y = {0:.4g} X1 + {1:.4g} X2 + {2:.4g}'.format(
    #     model.coef_[0], model.coef_[1], model.intercept_))
    # return model.coef_, model.intercept_

    print()
    # cm = metrics.confusion_matrix(y2, predictions)
    # print("Confusion Matrix : \n", cm)
    # print("Accuracy : ", metrics.accuracy_score(y2, predictions))

    calculateConfusionMatrix(y2, predictions,
                             [0, 1], printit=True)

    p = model.predict_proba(np.array([[60, 2, 199]]))
    # print(x2s.shape)
    # print(model.classes_)
    # print(np.array([[10,1,190]]).shape)
    # p = model.predict(np.array([[10, 1, 190]]))
    print(p)
    print()


def KNNExclusive(trainSet=[], testSet=[]):
    xs = []
    y = []
    for i in range(len(trainSet[0])):
        xs.append([trainSet[0][i], trainSet[1][i],
                  trainSet[2][i], trainSet[3][i]])
        y.append(trainSet[4][i])
    xs = np.array(xs)
    y = np.array(y)
    model = neighbors.KNeighborsClassifier(n_neighbors=5).fit(xs, y)

    x2s = []
    y2 = []
    for i in range(len(testSet[0])):
        x2s.append([testSet[0][i], testSet[1][i],
                   testSet[2][i], testSet[3][i]])
        y2.append(testSet[4][i])
    x2s = np.array(x2s)
    y2 = np.array(y2)
    predictions = model.predict(x2s)
    # y = a1x1 + a2x2 + b
    # print('Y = {0:.4g} X1 + {1:.4g} X2 + {2:.4g}'.format(
    #     model.coef_[0], model.coef_[1], model.intercept_))
    # return model.coef_, model.intercept_

    print()
    # cm = metrics.confusion_matrix(y2,predictions)
    # print("Confusion Matrix : \n", cm)
    # print("Accuracy : ", metrics.accuracy_score(y2, predictions))

    calculateConfusionMatrix(y2, predictions,
                             [0, 1], printit=True)

    p = model.predict_proba(np.array([[60, 2, 199, 0]]))
    print("Man: " + repr(p))
    p = model.predict_proba(np.array([[60, 2, 199, 1]]))
    print("Woman: " + repr(p))
    print()


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


def KMeansInclusive(dataset=[]):
    xs = []
    for i in range(len(dataset[0])):
        xs.append([dataset[0][i], dataset[1][i], dataset[2][i]])
    xs = np.array(xs)
    km = KMeans(k=2)
    km.fit(xs)
    print(km.centroids)
    pred = [km.pred(x) for x in xs]
    # print(pred)
    # print(km.classes)

    # print(model.labels_)
    # print(model.cluster_centers_)
    print()


def KMeansInclusive2(dataset=[]):
    xs = []
    for i in range(len(dataset[0])):
        xs.append([dataset[0][i], dataset[1][i], dataset[2][i]])
    xs = np.array(xs)
    model = cluster.KMeans(n_clusters=2).fit(xs)

    print(model.labels_)
    print(model.cluster_centers_)
    print()


if __name__ == "__main__":
    trainingPercentage = 0.2

    dataset = loadDataset('data/acath.csv', [], False)
    dataset = shuffleDataset(dataset)
    trainSet, testSet = divideDataset(dataset, trainingPercentage)

    # logisticRegressionInclusive(trainSet, testSet)
    # logisticRegressionExclusive(trainSet, testSet)
    # KNNInclusive(trainSet, testSet)
    # KNNExclusive(trainSet, testSet)
    KMeansInclusive(dataset)
    # KMeansInclusive2(dataset)
