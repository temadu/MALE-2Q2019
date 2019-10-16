import csv
import math
import random
import operator
import pandas as pd
import scipy.misc as smp
import cv2 as cv
import numpy as np
from PIL import Image


def loadTrainingFile(filepath, t):
    im = Image.open(filepath)
    set = []
    for pix in im.getdata():
        set.append([pix[0] / 256.0, pix[1] / 256.0, pix[2] / 256.0, t])
    return set


def loadTestingFileLabels(filepath):
    im = Image.open(filepath)
    set = []
    for pix in im.getdata():
        t = 2  # pasto
        if pix == 11:
            t = 1  # cielo
        elif pix == 9:
            t = 0  # vaca
        set.append(t)
    return set


def imageResult(predictions, kernel, c, rows, cols):
    # Create a 1024x1024x3 array of 8 bit unsigned integers
    data = np.zeros((cols, rows, 3), dtype=np.uint8)
    CLASS_NAMES = {0: "Linear", 1: "Poli", 2: "RBF", 3:"Sigmoid", 4:"Chi2", 5:"Inter"}

    for i in range(0, len(predictions)):
        if predictions[i] == 0:
            data[int(np.floor(i / rows)), int(i % rows)] = [165, 42, 42]
        elif predictions[i] == 1:
            data[int(np.floor(i / rows)), int(i % rows)] = [135, 206, 235]
        elif predictions[i] == 2:
            data[int(np.floor(i / rows)), int(i % rows)] = [124, 252, 0]

    img = Image.fromarray(data)  # Create a PIL image
    print(CLASS_NAMES[kernel] + str(c))
    img.show()


def calculateConfusionMatrix(testValues, predictions, posibleValues):
    allClasses = {}
    for i, c in enumerate(posibleValues):
        allClasses[c] = i
    confusionMatrix = [[0 for x in range(len(posibleValues))]
                       for y in range(len(posibleValues))]
    fp = 0
    fn = 0

    for i in range(len(testValues)):
        if predictions[i] == 1 and testValues[i] != 1:
            fp += 1
        if predictions[i] != 1 and testValues[i] == 1:
            fn += 1
        confusionMatrix[allClasses[testValues[i]]][allClasses[predictions[i]]] += 1
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
                                           (evaluationMeasurements[i][0] + evaluationMeasurements[i][1] +
                                            evaluationMeasurements[i][2] + evaluationMeasurements[i][3])
        except:
            evaluationMeasurements[i][4] = -1

        try:  # Precision tp/(fp+tp)
            evaluationMeasurements[i][5] = (
                                               evaluationMeasurements[i][0]) / (
                                                       evaluationMeasurements[i][0] + evaluationMeasurements[i][2])
        except:
            evaluationMeasurements[i][5] = -1

        try:  # Accuracy tp/(fp+tp)
            evaluationMeasurements[i][6] = (
                                               evaluationMeasurements[i][0]) / (
                                                       evaluationMeasurements[i][0] + evaluationMeasurements[i][3])
        except:
            evaluationMeasurements[i][6] = -1

        try:  # Accuracy tp/(fp+tp)
            evaluationMeasurements[i][7] = (
                                               evaluationMeasurements[i][1]) / (
                                                       evaluationMeasurements[i][1] + evaluationMeasurements[i][2])
        except:
            evaluationMeasurements[i][7] = -1

        try:
            evaluationMeasurements[i][8] = (2 * evaluationMeasurements[i][0]) / (
                    2 * evaluationMeasurements[i][0] + evaluationMeasurements[i][2] + evaluationMeasurements[i][3])
        except:
            evaluationMeasurements[i][8] = -1

    print("EVALUATION MEASUREMENTS")
    pd.set_option('display.expand_frame_repr', False)
    print(pd.DataFrame(evaluationMeasurements,
                       columns=evaluationMeasurementsTitles, index=evaluationMeasurementsRowNames))
    print()


def main():
    mode = True
    cows = True
    trainingSet = [[], []]
    testSet = [[], []]

    if mode:
        cowSet = loadTrainingFile('data/images/vaca.jpg', 0)
        skySet = loadTrainingFile('data/images/cielo.jpg', 1)
        grassSet = loadTrainingFile('data/images/pasto.jpg', 2)
        comb = cowSet + skySet + grassSet
        random.shuffle(comb)
        split = 0.6
        trainingSet[0].extend(comb[:math.floor(len(comb) * split)])
        testSet[0].extend(comb[math.floor(len(comb) * split):])

        for row in trainingSet[0]:
            trainingSet[1].append(row.pop(-1))
        for row in testSet[0]:
            testSet[1].append(row.pop(-1))
    else:
        set = [[], []]
        set[0] = loadTrainingFile('data/images/cow.jpg', -1)
        set[1] = loadTestingFileLabels('data/images/cow16.bmp')
        for row, label in zip(set[0], set[1]):
            row[3] = label
        split = 0.6
        random.shuffle(set[0])
        trainingSet[0].extend(set[0][:math.floor(len(set[0]) * split)])
        testSet[0].extend(set[0][math.floor(len(set[0]) * split):])

        for row in trainingSet[0]:
            trainingSet[1].append(row.pop(-1))
        for row in testSet[0]:
            testSet[1].append(row.pop(-1))


    print('Train set: ' + repr(len(trainingSet[0])))
    print('Test set: ' + repr(len(testSet[0])))

    # Train the SVM
    trainingData = np.matrix(trainingSet[0], dtype=np.float32)
    trainingLabels = np.array(trainingSet[1])

    CLASS_NAMES = {0: "Linear", 1: "Poli", 2: "RBF", 3:"Sigmoid", 4:"Chi2", 5:"Inter"}

    for a in range(0,6):
        for c in [0.1, 0.5, 1, 5, 10]:
            print(CLASS_NAMES[a])
            print(c)
            svm = cv.ml.SVM_create()
            svm.setType(cv.ml.SVM_C_SVC)
            svm.setC(c)
            svm.setDegree(3)
            svm.setGamma(1)
            svm.setKernel(a)
            svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
            svm.train(trainingData, cv.ml.ROW_SAMPLE, trainingLabels)

            # Predict
            testingData = np.matrix(testSet[0], dtype=np.float32)
            # testingLabels = np.array(testSet[1])

            sv = svm.getUncompressedSupportVectors()
            predictions = svm.predict(testingData)[1]
            predictions = list(predictions.transpose()[0])  # Transform vertical nparray to list

            calculateConfusionMatrix(testSet[1], predictions, [0, 1, 2])

            if cows:
                cowPredict = loadTrainingFile('data/images/cow.jpg', -1)
                for row in cowPredict:
                    row.pop(-1)
                testingData = np.matrix(cowPredict, dtype=np.float32)
                predictions = svm.predict(testingData)[1]
                imageResult(predictions, a, c, 1140, 760)

                cowPredict2 = loadTrainingFile('data/images/vaca2.jpg', -1)
                for row in cowPredict2:
                    row.pop(-1)
                testingData2 = np.matrix(cowPredict2, dtype=np.float32)
                predictions2 = svm.predict(testingData2)[1]
                imageResult(predictions2, a, c, 600, 360)


main()
