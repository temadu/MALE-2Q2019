import math
import re
import random
from collections import defaultdict
import unidecode
from fractions import Fraction
import numpy as np
import pandas as pd

def getWordsInLowerCase(text):
    assert type(text) in (str, bytes)
    words = re.findall("[a-z]{2,}", text, re.I)
    words = map(lambda x: unidecode.unidecode(x).lower(), words)
    return [word for word in words]


def wordFreq(text):
    words = getWordsInLowerCase(text)
    count = defaultdict(int)
    for word in words:
        count[word] += 1
    return count


file = open('data/noticias_argentinas_aa_bayes.tsv', "r", encoding="utf8")
trainPercentage = 0.8


data = []
trainData = []
testData = []
emptyWords = [
  "a",
  "acuerdo",
  "adelante",
  "ademas",
  "adrede",
  "ahi",
  "ahora",
  "al",
  "alli",
  "alrededor",
  "antano",
  "ante",
  "antes",
  "apenas",
  "aproximadamente",
  "aquel",
  "aquella",
  "aquellas",
  "aquello",
  "aquellos",
  "aqui",
  "arriba",
  "abajo",
  "asi",
  "aun",
  "aunque",
  "b",
  "bajo",
  "bastante",
  "bien",
  "breve",
  "c",
  "casi",
  "cerca",
  "claro",
  "como",
  "con",
  "conmigo",
  "contigo",
  "contra",
  "cual",
  "cuales",
  "cuando",
  "cuanta",
  "cuantas",
  "cuanto",
  "cuantos",
  "d",
  "de",
  "debajo",
  "del",
  "delante",
  "demasiado",
  "dentro",
  "deprisa",
  "desde",
  "despacio",
  "despues",
  "detras",
  "dia",
  "dias",
  "donde",
  "dos",
  "durante",
  "e",
  "el",
  "ella",
  "ellas",
  "ellos",
  "en",
  "encima",
  "enfrente",
  "enseguida",
  "entre",
  "es",
  "esa",
  "esas",
  "ese",
  "eso",
  "esos",
  "esta",
  "estado",
  "estados",
  "estan",
  "estar",
  "estas",
  "este",
  "esto",
  "estos",
  "ex",
  "excepto",
  "f",
  "final",
  "fue",
  "fuera",
  "fueron",
  "g",
  "general",
  "gran",
  "h",
  "ha",
  "habia",
  "habla",
  "hablan",
  "hace",
  "hacia",
  "han",
  "hasta",
  "hay",
  "horas",
  "hoy",
  "i",
  "incluso",
  "informo",
  "j",
  "junto",
  "k",
  "l",
  "la",
  "lado",
  "las",
  "le",
  "lejos",
  "lo",
  "los",
  "luego",
  "m",
  "mal",
  "mas",
  "mayor",
  "me",
  "medio",
  "mejor",
  "menos",
  "menudo",
  "mi",
  "mia",
  "mias",
  "mientras",
  "mio",
  "mios",
  "mis",
  "mismo",
  "mucho",
  "muy",
  "n",
  "nada",
  "nadie",
  "ninguna",
  "no",
  "nos",
  "nosotras",
  "nosotros",
  "nuestra",
  "nuestras",
  "nuestro",
  "nuestros",
  "nueva",
  "nuevo",
  "nunca",
  "o",
  "os",
  "otra",
  "otros",
  "p",
  "pais",
  "para",
  "parte",
  "pasado",
  "peor",
  "pero",
  "poco",
  "por",
  "porque",
  "pronto",
  "proximo",
  "puede",
  "q",
  "qeu",
  "que",
  "quien",
  "quienes",
  "quiza",
  "quizas",
  "r",
  "raras",
  "repente",
  "s",
  "salvo",
  "se",
  "segun",
  "ser",
  "sera",
  "si",
  "sido",
  "siempre",
  "sin",
  "sobre",
  "solamente",
  "solo",
  "son",
  "soyos",
  "su",
  "supuesto",
  "sus",
  "suya",
  "suyas",
  "suyo",
  "t",
  "tal",
  "tambien",
  "tampoco",
  "tarde",
  "te",
  "temprano",
  "ti",
  "tiene",
  "todavia",
  "todo",
  "todos",
  "tras",
  "tu",
  "tus",
  "tuya",
  "tuyas",
  "tuyo",
  "tuyos",
  "u",
  "un",
  "una",
  "unas",
  "uno",
  "unos",
  "usted",
  "ustedes",
  "v",
  "veces",
  "vez",
  "vosotras",
  "vosotros",
  "vuestra",
  "vuestras",
  "vuestro",
  "vuestros",
  "w",
  "x",
  "y",
  "ya",
  "yo",
  "z"
]

classes = defaultdict(lambda: defaultdict(int))
file.readline()
for line in file:
    className = line.split("\t")[3]
    if(className == "Noticias destacadas\n" or className == "Destacadas\n"):
        continue
    data.append(line)
file.close()
print("Total de tuplas: ", len(data))

# Separate data into trainData and testData
random.shuffle(data)
trainData = data[:math.floor(len(data)*trainPercentage)]
# trainData = data[:2000]
testData = data[math.floor(len(data)*trainPercentage):]
# testData = data[-2000:]

# train
print("START TRAINING: ", len(trainData))
for i, t in enumerate(trainData):
    # print(i, end="\r")
    row = t.split("\t")
    title = row[1]
    className = row[3][:-1]
    frequents = wordFreq(title)
    # print("%s: %s " % (className, title))
    # print(frequents)
    for word, freq in frequents.items():
        classes[className][word] += freq

allClasses = {}
for i, c in enumerate(classes.keys()):
  allClasses[c] = i  
confusionMatrix = [[0 for x in range(len(allClasses))]
                   for y in range(len(allClasses))]

# test
print("START TESTING: ", len(testData))
probs = {}
for i, t in enumerate(testData):
    # print(i, end="\r")
    row = t.split("\t")
    title = row[1]
    realClassName = row[3][:-1]
    # print("%s: %s " % (realClassName, title))
    words = getWordsInLowerCase(title)  # Separo en palabras
    for className in classes:
        numWords = sum(classes[className].values()) #Cuantas palabras totales hay en la clase
        logSum = 0
        for word in words:
            wordFreq = classes[className][word] + 1
            prob = Fraction(wordFreq, numWords+len(allClasses))
            logSum += math.log(prob)        #Uso log y sumo xq usa probabilidades muy bajas y si multiplico se va todo al demonio x los floating points
        # probs[className] = logSum
        probs[className] = logSum
    # print(probs)
    highestClasses = sorted(probs, key=probs.get)
    predictedClassName = highestClasses[-1]
    highest = probs[predictedClassName]
    # print("%s: %f " % (highestClasses[-1], highest))
    # print()

    confusionMatrix[allClasses[realClassName]][allClasses[predictedClassName]]+=1
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
      elif i == j:
        evaluationMeasurements[i][3] += confusionMatrix[j][k]
        evaluationMeasurements[len(allClasses)][3] += confusionMatrix[j][k]
      elif i == k:
        evaluationMeasurements[i][2] += confusionMatrix[j][k]
        evaluationMeasurements[len(allClasses)][2] += confusionMatrix[j][k]
      else:
        evaluationMeasurements[i][1] += confusionMatrix[j][k]
        evaluationMeasurements[len(allClasses)][1] += confusionMatrix[j][k]

  evaluationMeasurements[i][4] = (evaluationMeasurements[i][0] + \
                                  evaluationMeasurements[i][1])/(evaluationMeasurements[i][0]+evaluationMeasurements[i][1]+evaluationMeasurements[i][2]+evaluationMeasurements[i][3])
  evaluationMeasurements[i][5] = (evaluationMeasurements[i][0])/(evaluationMeasurements[i][0]+evaluationMeasurements[i][2])

  evaluationMeasurements[i][6] = (evaluationMeasurements[i][0])/(evaluationMeasurements[i][0]+evaluationMeasurements[i][3])
  evaluationMeasurements[i][7] = (evaluationMeasurements[i][1])/(evaluationMeasurements[i][1]+evaluationMeasurements[i][2])
  
  evaluationMeasurements[i][8] = (2*evaluationMeasurements[i][0])/(2*evaluationMeasurements[i][0]+evaluationMeasurements[i][2]+evaluationMeasurements[i][3])


      


print("CONFUSION MATRIX")
print(pd.DataFrame(confusionMatrix, columns=list(
    allClasses.keys()), index=list(allClasses.keys())))
print()

print("EVALUATION MEASUREMENTS")
print(pd.DataFrame(evaluationMeasurements,
                   columns=evaluationMeasurementsTitles, index=evaluationMeasurementsRowNames))










    # total = 0
    # for n in probs:
    #   total += probs[n]
    #   print("%s: %f " % (n, probs[n]))
    # print()
    # print("Total:", total, "=", float(total))
