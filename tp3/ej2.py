import csv
import math
import random
import operator
import pandas as pd
import cv2 as cv
import numpy as np

def loadDataset(filename, split, trainingSet=[], testSet=[]):
  with open(filename, encoding="utf8") as csvfile:
      reader = csv.reader(csvfile, delimiter=",")
      first = True
      dataset = []
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
        tupl.append(float(row[0])) # age
        tupl.append(float(row[1])) # dias sintiendose mal
        tupl.append(float(row[2]))  # colesterol
        tupl.append(int(row[3])) # sex 0 o 1 
        tupl.append(int(row[5]))  # Tres arterias estrechadas
        tupl.append(int(row[4]))  # Una arteria estrechada (lo que busco)
        dataset.append(tupl)
      
      print("max age: " + str(maxAge))
      print("max days: " + str(maxBadDays))
      print("max cholesterol: " + str(maxCholesterol))

      for row in dataset:
        row[0] = row[0]/maxAge
        row[1] = row[1]/maxBadDays
        row[2] = row[2]/maxCholesterol

      random.shuffle(dataset)
      trainingSet.append([])
      trainingSet.append([])
      testSet.append([])
      testSet.append([])
      trainingSet[0].extend(dataset[:math.floor(len(dataset)*split)])
      testSet[0].extend(dataset[math.floor(len(dataset)*split):])
      
      for row in trainingSet[0]:
        trainingSet[1].append(row.pop(-1))
      for row in testSet[0]:
        testSet[1].append(row.pop(-1))


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
    print(pd.DataFrame(confusionMatrix, columns=posibleValues, index=posibleValues))
    print()

  return calculateStatistics(posibleValues, confusionMatrix, printit)


def calculateStatistics(posibleValues, confusionMatrix, printit = True):
  allClasses = {}
  for i, c in enumerate(posibleValues):
    allClasses[c] = i

  evaluationMeasurementsRowNames = list(allClasses.keys()).copy()
  evaluationMeasurementsRowNames.append("TOTAL")
  evaluationMeasurementsTitles = [
      "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1Score"]
  evaluationMeasurements = [[0 for x in range(len(
      evaluationMeasurementsTitles))] for y in range(len(evaluationMeasurementsRowNames))]
  for i in range(len(evaluationMeasurementsRowNames)):
    for j in range(len(allClasses)):
      for k in range(len(allClasses)):
        if i == j and i == k:
          evaluationMeasurements[i][0] = confusionMatrix[j][k]
          evaluationMeasurements[len(allClasses)][0] += confusionMatrix[j][k]
        elif i == k:
          evaluationMeasurements[i][3] += confusionMatrix[j][k]
          evaluationMeasurements[len(allClasses)][3] += confusionMatrix[j][k]
        elif i == j:
          evaluationMeasurements[i][2] += confusionMatrix[j][k]
          evaluationMeasurements[len(allClasses)][2] += confusionMatrix[j][k]
        else:
          evaluationMeasurements[i][1] += confusionMatrix[j][k]
          evaluationMeasurements[len(allClasses)][1] += confusionMatrix[j][k]

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

def getBestConfig(trainingSet, testSet):
  trainingData = np.matrix(trainingSet[0], dtype=np.float32)
  trainingLabels = np.array(trainingSet[1])
  testingData = np.matrix(testSet[0], dtype=np.float32)
  bestAc = 0
  bestC = 0
  bestK = 0
  for k in range(3):  # Linear, poly, rbf, sigmoid
    print()
    for c in np.arange(0.1, 10.2, 0.5):
      accuracies = 0
      # for i in range(10):
      svm = cv.ml.SVM_create()
      svm.setType(cv.ml.SVM_C_SVC)
      svm.setKernel(k)
      svm.setC(c)
      svm.setDegree(4)
      svm.setTermCriteria(
          (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
      svm.train(trainingData, cv.ml.ROW_SAMPLE, trainingLabels)

      # Predict
      # testingLabels = np.array(testSet[1])

      sv = svm.getUncompressedSupportVectors()
      predictions = svm.predict(testingData)[1]
      # Transform vertical nparray to list
      predictions = list(predictions.transpose()[0])

      accuracy = calculateConfusionMatrix(
          testSet[1], predictions, [0, 1], False)
      if(accuracy > bestAc):
        bestAc = accuracy
        bestC = c
        bestK = k
      if k == 0:
        print('KERNEL: LINEAR, ' + "C = " + str(c) + " -> ac: "+str(accuracy))
      elif k == 1:
        print('KERNEL: POLY, ' + "C = " + str(c) + " -> ac: "+str(accuracy))
      elif k == 2:
        print('KERNEL: RBF, ' + "C = " + str(c) + " -> ac: "+str(accuracy))

  print()
  if bestK == 0:
    print('BEST CONFIG: KERNEL: LINEAR, ' +
          "C = " + str(bestC) + " -> Accuracy: "+str(bestAc))
  elif bestK == 1:
    print('BEST CONFIG: KERNEL: POLY, ' + "C = " +
          str(bestC) + " -> Accuracy: "+str(bestAc))
  elif bestK == 2:
    print('BEST CONFIG: KERNEL: RBF, ' + "C = " +
          str(bestC) + " -> Accuracy: "+str(bestAc))

def trainOnce(trainingSet, testSet, c, k):
  # Train the SVM
  trainingData = np.matrix(trainingSet[0], dtype=np.float32)
  trainingLabels = np.array(trainingSet[1])
  testingData = np.matrix(testSet[0], dtype=np.float32)

  svm = cv.ml.SVM_create()
  svm.setType(cv.ml.SVM_C_SVC)
  svm.setKernel(k)
  svm.setC(c)
  svm.setDegree(4)
  svm.setTermCriteria(
      (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
  svm.train(trainingData, cv.ml.ROW_SAMPLE, trainingLabels)

  # Predict
  # testingLabels = np.array(testSet[1])

  sv = svm.getUncompressedSupportVectors()
  predictions = svm.predict(testingData)[1]
  # Transform vertical nparray to list
  predictions = list(predictions.transpose()[0])

  if k == 0:
    print('KERNEL: LINEAR, ' + "C = " + str(c) )
  elif k == 1:
    print('KERNEL: POLY, ' + "C = " + str(c) )
  elif k == 2:
    print('KERNEL: RBF, ' + "C = " + str(c) )
  accuracy = calculateConfusionMatrix(testSet[1], predictions, [0, 1], True)

def main():
  trainingPercentage = 0.5

  trainingSet = []
  testSet = []
  loadDataset('data/acath.csv',
              trainingPercentage, trainingSet, testSet)
  print('Train set: ' + repr(len(trainingSet[0])))
  print('Test set: ' + repr(len(testSet[0])))

  getBestConfig(trainingSet, testSet)
  # trainOnce(trainingSet,testSet, 5.1, 2)


  

main()
