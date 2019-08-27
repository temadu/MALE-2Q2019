import math
import re
import random
from collections import defaultdict
import unidecode
from fractions import Fraction


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
trainPercentage = 0.5
data = []
trainData = []
testData = []

classes = defaultdict(lambda: defaultdict(int))
file.readline()
for line in file:
    data.append(line)
# data = file.read().split("\n")
file.close()

# Separate data into trainData and testData
random.shuffle(data)
trainData = data[:math.floor(len(data)*trainPercentage)]
# trainData = data[:5]
# testData = data[math.floor(len(data)*trainPercentage):]
testData = data[-3:]

# train
for i, t in enumerate(trainData):
    row = t.split("\t")
    title = row[1]
    className = row[3][:-1]
    frequents = wordFreq(title)
    # print("%s: %s " % (className, title))
    # print(frequents)
    for word, freq in frequents.items():
        classes[className][word] += freq
    print(i, end="\r")

print("FINISHED TRAINING")

# test
probs = {}
for t in testData:
    row = t.split("\t")
    title = row[1]
    realClassName = row[3][:-1]
    print("%s: %s " % (realClassName, title))
    for c in classes:
        words = getWordsInLowerCase(title)
        numWords = sum(classes[c].values())
        logSum = 0
        for word in words:
            freq = classes[c][word] + 1
            prob = Fraction(freq, numWords)
            logSum += math.log(prob)
        probs[c] = logSum
    # print(probs)
    highestClass = sorted(probs, key=probs.get)[-1]
    highest = probs[highestClass]
    print("%s: %f " % (highestClass, highest))
    print()
