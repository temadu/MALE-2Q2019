import csv
import math
import random
import operator
import pandas as pd



def calculatePromWordsInStars(filename, stars):
  with open(filename, encoding="utf8") as csvfile:
      reader = csv.reader(csvfile, delimiter=";")
      first = True
      s = 0
      c = 0
      for row in reader:
        if first:
          first = False
          continue
        # print(row)
        if(float(row[5]) == stars):
          s += float(row[2])
          c += 1
      print("Promedio de palabras con ", stars, " estrellas: ", (s/c))  

def loadDataset(filename, split, trainingSet=[], testSet=[]):
  with open(filename, encoding="utf8") as csvfile:
      reader = csv.reader(csvfile, delimiter=";")
      first = True
      dataset = []
      highestWordCount = 0
      for row in reader:
        if first:
          first = False
          continue
        # print(row)
        tupl = []
        tupl.append(float(row[2]))
        if(float(row[2]) > highestWordCount):
          highestWordCount = float(row[2])
        # tupl.append(1 if row[3] == "positive" else 0)
        tupl.append(1 if row[3] == "positive" else (0 if row[3] == "negative" else 0.5))
        tupl.append(float(row[6])/8+0.5) 
        tupl.append(float(row[5]))
        dataset.append(tupl)
      
      for row in dataset:
        row[0] = row[0]/highestWordCount

      random.shuffle(dataset)
      trainingSet.extend(dataset[:math.floor(len(dataset)*split)])
      testSet.extend(dataset[math.floor(len(dataset)*split):])


def euclideanDistance(instance1, instance2, length):
  distance = 0
  for x in range(length):
    distance += pow((instance1[x] - instance2[x]), 2)
  return math.sqrt(distance)


def getNeighborsWithDistances(trainingSet, testInstance):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	return distances


def getResponse(distances,  k):
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(),
	                     key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getWeightedResponse(distances, k):
  neighbors = []
  if(k == 0):
    for x in range(len(distances)):
        neighbors.append(distances[x])
  else:
    for x in range(k):
        neighbors.append(distances[x])
  classVotes = {}
  for x in range(len(neighbors)):
    response = neighbors[x][0][-1]
    if response in classVotes:
      if(pow(neighbors[x][1], 2) != 0):
        classVotes[response] += 1/pow(neighbors[x][1], 2)
      else:
        classVotes[response] += 1/0.00000000001
    else:
      if(pow(neighbors[x][1], 2) != 0):
        classVotes[response] = 1/pow(neighbors[x][1], 2)
      else:
        classVotes[response] = 1/0.00000000001
  sortedVotes = sorted(classVotes.items(),
	                     key=operator.itemgetter(1), reverse=True)
  return sortedVotes[0][0]

def calculateConfusionMatrix(testSet, predictions, posibleValues):
  allClasses = {}
  for i, c in enumerate(posibleValues):
    allClasses[c] = i
  confusionMatrix = [[0 for x in range(len(posibleValues))]
                     for y in range(len(posibleValues))]
  fp = 0
  fn = 0

  for i in range(len(testSet)):
    if(predictions[i] == 1 and testSet[i][-1] != 1):
      fp += 1
    if(predictions[i] != 1 and testSet[i][-1] == 1):
      fn += 1
    confusionMatrix[allClasses[predictions[i]]
                    ][allClasses[testSet[i][-1]]] += 1
  print("CONFUSION MATRIX")
  print(pd.DataFrame(confusionMatrix, columns=posibleValues, index=posibleValues))
  print()

  calculateStatistics(posibleValues, confusionMatrix)

def calculateStatistics(posibleValues, confusionMatrix):
  allClasses = {}
  for i, c in enumerate(posibleValues):
    allClasses[c] = i

  evaluationMeasurementsRowNames = list(allClasses.keys()).copy()
  evaluationMeasurementsRowNames.append("TOTAL")
  evaluationMeasurementsTitles = ["TP","TN","FP", "FN", "Accuracy", "Precision","Sensitivity", "Specificity", "F1Score"]
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

    try: # Accuracy tp+tn/all
      evaluationMeasurements[i][4] = (evaluationMeasurements[i][0] + evaluationMeasurements[i][1]) / \
      (evaluationMeasurements[i][0]+evaluationMeasurements[i][1]+evaluationMeasurements[i][2]+evaluationMeasurements[i][3])
    except:
      evaluationMeasurements[i][4] = -1

    try:  # Precision tp/(fp+tp)
      evaluationMeasurements[i][5] = (evaluationMeasurements[i][0])/(evaluationMeasurements[i][0]+evaluationMeasurements[i][2])
    except:
      evaluationMeasurements[i][5] = -1
    
    try:  # Accuracy tp/(fp+tp)
      evaluationMeasurements[i][6] = (evaluationMeasurements[i][0])/(evaluationMeasurements[i][0]+evaluationMeasurements[i][3])
    except:
      evaluationMeasurements[i][6] = -1
    
    try:  # Accuracy tp/(fp+tp)
      evaluationMeasurements[i][7] = (evaluationMeasurements[i][1])/(evaluationMeasurements[i][1]+evaluationMeasurements[i][2])
    except:
      evaluationMeasurements[i][7] = -1

    try:
      evaluationMeasurements[i][8] = (2*evaluationMeasurements[i][0])/(2*evaluationMeasurements[i][0]+evaluationMeasurements[i][2]+evaluationMeasurements[i][3])
    except:
      evaluationMeasurements[i][8] = -1

  print("EVALUATION MEASUREMENTS")
  print(pd.DataFrame(evaluationMeasurements,
                   columns=evaluationMeasurementsTitles, index=evaluationMeasurementsRowNames))
  print()

def prom():
  calculatePromWordsInStars('data/reviews_sentiment.csv', 1.0)
  calculatePromWordsInStars('data/reviews_sentiment.csv', 2.0)
  calculatePromWordsInStars('data/reviews_sentiment.csv', 3.0)
  calculatePromWordsInStars('data/reviews_sentiment.csv', 4.0)
  calculatePromWordsInStars('data/reviews_sentiment.csv', 5.0)
  print()

def main():
  trainingPercentage = 0.7
  k = 5
  weighted = True
  parta = True

  if(parta):
    prom()
  
  trainingSet = []
  testSet = []
  loadDataset('data/reviews_sentiment.csv',
              trainingPercentage, trainingSet, testSet)
  print('Train set: ' + repr(len(trainingSet)))
  print('Test set: ' + repr(len(testSet)))

  predictions = []
  for x in range(len(testSet)):
    neighbors = getNeighborsWithDistances(trainingSet, testSet[x])
    if(weighted):
      result = getWeightedResponse(neighbors, k)
    else:
      result = getResponse(neighbors, k)
    predictions.append(result)

  calculateConfusionMatrix(testSet, predictions, [1,2,3,4,5])


main()
