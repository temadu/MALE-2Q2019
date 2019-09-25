import csv
import math
import anytree
from anytree.exporter import DotExporter
def gain(s, att, node, h_s):
    # por ej s = females, att = SEX_ATTS
    # print("{} {} {} {}".format(len(s), att, node, h_s))
    gain = h_s
    entropies = {}
    for a in att:
        attEntropy = 0
        attArr = [x for x in s if x[node] == a]
        attPos = sum(1.0 if x[0] else 0.0 for x in attArr)
        attNeg = sum(1.0 if not x[0] else 0.0 for x in attArr)
        attTot = attPos + attNeg
        if attPos != 0 and attNeg != 0:
            attEntropy = -(attPos/attTot) * math.log2(attPos/attTot) - (attNeg/attTot) * math.log2(attNeg/attTot)
        elif attPos != 0:
            attEntropy = -(attPos / attTot) * math.log2(attPos / attTot)
        elif attNeg != 0:
            attEntropy = - (attNeg / attTot) * math.log2(attNeg / attTot)
        entropies[a] = attEntropy
        gain -= (attTot/len(s)) * attEntropy
    return [gain, entropies]


# passenger: bool surv, int class, bool male, float age
names = {1: "CLASS", 2: "SEX", 3: "AGE"}
numbers = {"CLASS": 1, "SEX": 2, "AGE": 3}
CLASS = 1
SEX = 2
AGE = 3
passengers = []
CLASS_ATTS = [1, 2, 3]
SEX_ATTS = [True, False]
AGE_ATTS = [-1, 0, 1]
CLASS_NAMES = {1: "First Class", 2: "Second Class", 3:"Third Class"}
SEX_NAMES = {True: "Male", False: "Female"}
AGE_NAMES = {1: "Kid", 0: "Adult", -1: "N/A"}
nodes_names = [CLASS_NAMES, SEX_NAMES, AGE_NAMES]
nodes_atts = [CLASS_ATTS, SEX_ATTS, AGE_ATTS]
nodes = [CLASS, SEX, AGE]

root = ''
with open('./data/titanic.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    first = True
    # print(n)
    for row in reader:
        if first:
            first = False
            continue
        passenger = [0, 0, 0, 0]
        passenger[0] = row[1] == "1"
        passenger[1] = int(row[2])
        passenger[2] = row[4] == "male"
        if row[5] == "":
            passenger[3] = -1
        else:
            passenger[3] = 1 if float(row[5]) <= 14.0 else 0
        passengers.append(passenger)
csvFile.close()

# print(passengers)

# full entropy
posT = sum(1.0 if x[0] else 0.0 for x in passengers)
negT = sum(1.0 if not x[0] else 0.0 for x in passengers)
totT = posT + negT

fullEntropy = -posT/totT * math.log2(posT/totT) - negT/totT * math.log2(negT/totT)

g = gain(passengers, AGE_ATTS, AGE, fullEntropy)
gain_S_age = g[0]
age_entropies = g[1]
# print(gain_S_age)

g = gain(passengers, CLASS_ATTS, CLASS, fullEntropy)
gain_S_class = g[0]
class_entropies = g[1]

# print(gain_S_class)

g = gain(passengers, SEX_ATTS, SEX, fullEntropy)
gain_S_sex = g[0]
sex_entropies = g[1]

# print(gain_S_sex)

entropies = {}
if gain_S_class > gain_S_age and gain_S_class > gain_S_sex:
    root = CLASS
    rootNode = anytree.Node("CLASS")
    for att in nodes_atts[CLASS - 1]:
        name = CLASS_NAMES[att]
        anytree.Node(name, parent=rootNode, s=[x for x in passengers if x[root] == att], att=att, nodes=[SEX, AGE], entropies=class_entropies)
    # nodes = [SEX, AGE]
    # entropies = class_entropies
elif gain_S_age > gain_S_sex and gain_S_age > gain_S_class:
    rootNode = anytree.Node("AGE")
    root= AGE
    for att in nodes_atts[AGE - 1]:
        name = AGE_NAMES[att]
        anytree.Node(name, parent=rootNode, s=[x for x in passengers if x[root] == att], att=att, nodes=[SEX, CLASS], entropies=age_entropies)

    # nodes = [SEX, CLASS]
    # entropies = age_entropies
else:
    root = SEX
    rootNode = anytree.Node("SEX")
    for att in nodes_atts[SEX - 1]:
        name = SEX_NAMES[att]
        anytree.Node(name, parent=rootNode, s=[x for x in passengers if x[root] == att], att=att, nodes=[CLASS, AGE], entropies=sex_entropies)
    # nodes = [CLASS, AGE]
    # entropies = sex_entropies

print(rootNode.leaves[1])

# DotExporter(rootNode).to_picture("./caca.png")
for pre, fill, node in anytree.RenderTree(rootNode):
    print("%s%s" % (pre, node.name))
stop = False
while(not stop):
    for i in rootNode.leaves:
        print(i)
    for treeNode in rootNode.leaves:
        maxGain = 0
        maxEntropies = {}
        node = -1
        for attribute in treeNode.nodes:
            s = treeNode.s
            # print("{} {}".format(treeNode, attribute))
            g = gain(s, nodes_atts[attribute - 1], attribute, treeNode.entropies[treeNode.att])
            if g[0] > maxGain:
                maxGain = g[0]
                maxEntropies = g[1]
                node = attribute
        # print(i)
        # print(maxGain)
        # print(names[node])
        print("el nodo discriminante es " + names[node])
        parent = treeNode
        newNodes = [x for x in treeNode.nodes if x != node]
        print(newNodes)
        if len(newNodes) == 0:
            stop = True
        newNode = anytree.Node(names[node], parent=parent, nodes=newNodes)
        for value in nodes_atts[node - 1]:
            name = nodes_names[node - 1][value]
            anytree.Node(name, parent=newNode, nodes=newNodes, s=[x for x in treeNode.s if x[node] == value], att=value, entropies=maxEntropies)

    for pre, fill, node in anytree.RenderTree(rootNode):
        print("%s%s" % (pre, node.name))


for node in rootNode.leaves:
    print(node.parent.parent.parent.parent.name)
    print(node.parent.parent.name)
    print(node.name)
    # print(node.s)
    surv = sum(1.0 if x[0] else 0.0 for x in node.s)
    notSurv = sum(1.0 if not x[0] else 0.0 for x in node.s)
    anytree.Node("Survived" if surv > notSurv else "Died", parent=node)



for pre, fill, node in anytree.RenderTree(rootNode):
    print("%s%s" % (pre, node.name))
