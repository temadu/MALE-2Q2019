import csv
import math
import anytree
import random
import matplotlib.pyplot as plt
import numpy as np


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


def gainGini(s, att, node, h_s):
    gain = h_s
    entropies = {}
    for a in att:
        attEntropy = 0
        attArr = [x for x in s if x[node] == a]
        attPos = sum(1.0 if x[0] else 0.0 for x in attArr)
        attNeg = sum(1.0 if not x[0] else 0.0 for x in attArr)
        attTot = attPos + attNeg
        attEntropy = 1 - (attPos/attTot)**2 - (attNeg/attTot)**2
        entropies[a] = attEntropy
        gain -= (attTot / len(s)) * attEntropy
    return [gain, entropies]


def test(p, root):
    node = root
    while not node.is_leaf:
        for c in node.children:
            if p[numbers[node.display_name]] == c.att:
                node = c.children[0]
                break
    return node.att


def calc(rootNode, testing, training, matrix):
    surv = 0
    correct = 0
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    # for p in testing:
    for p in testing:
        res = test(p, rootNode)
        if res and p[0]:
            truePos += 1
        elif not res and not p[0]:
            trueNeg += 1
        elif res and not p[0]:
            falsePos += 1
        else:
            falseNeg += 1
        surv += res
        correct += res == p[0]
    if matrix:
        print("\t\tPredictions")
        print("\t\tSurv\tDied")
        print("Surv\t" + str(truePos) + "\t\t" + str(falseNeg))
        print("Died\t" + str(falsePos) + "\t\t" + str(trueNeg))
    testingPrec = correct/len(testing)
    correct = 0

    for p in training:
        res = test(p, rootNode)
        correct += res == p[0]
    trainingPrec = correct / len(training)
    return[testingPrec, trainingPrec]


def forestCalc(rootNodes):
    correct = 0
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for p in testing:
        res = 0
        for rn in rootNodes:
            res += test(p, rn[0])
        if res > len(rootNodes)/2:
            res = True
        else:
            res = False
        if res and p[0]:
            truePos += 1
        elif not res and not p[0]:
            trueNeg += 1
        elif res and not p[0]:
            falsePos += 1
        else:
            falseNeg += 1
        correct += res == p[0]
    print("\t\tPredictions")
    print("\t\tSurv\tDied")
    print("Surv\t" + str(truePos) + "\t\t" + str(falseNeg))
    print("Died\t" + str(falsePos) + "\t\t" + str(trueNeg))
    print(correct / len(testing))


# passenger: bool surv, int class, bool male, float age
gini = True
# nodeid = 0
names = {1: "CLASS", 2: "SEX", 3: "AGE"}
numbers = {"CLASS": 1, "SEX": 2, "AGE": 3}
CLASS = 1
SEX = 2
AGE = 3
testing = []
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
        testing.append(passenger)
csvFile.close()


def run():

    trainingPrec = []
    testingPrec = []
    x_nodes = []
    #training \/
    nodeid = 0
    passengers = random.choices(testing, k=int(len(testing) * 0.1))

    # print(passengers)

    # full entropy
    posT = sum(1.0 if x[0] else 0.0 for x in passengers)
    negT = sum(1.0 if not x[0] else 0.0 for x in passengers)
    totT = posT + negT

    if gini:
        fullEntropy = -posT/totT * math.log2(posT/totT) - negT/totT * math.log2(negT/totT)
    else:
        fullEntropy = 1 - (posT/totT)**2 - (negT/totT)**2

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
        nodeid += 1
        rootNode = anytree.Node(nodeid, display_name="CLASS", is_class=True)
        for att in nodes_atts[CLASS - 1]:
            name = CLASS_NAMES[att]
            nodeid += 1
            aux = anytree.Node(nodeid, display_name=name, is_class=False, parent=rootNode, s=[x for x in passengers if x[root] == att], att=att, nodes=[SEX, AGE], entropies=class_entropies)
            surv = sum(1.0 if x[0] else 0.0 for x in aux.s)
            notSurv = sum(1.0 if not x[0] else 0.0 for x in aux.s)
            anytree.Node(-1, parent=aux, att=surv>notSurv)
        # nodes = [SEX, AGE]
        # entropies = class_entropies
    elif gain_S_age > gain_S_sex and gain_S_age > gain_S_class:
        nodeid += 1
        rootNode = anytree.Node(nodeid, display_name="AGE", is_class=True)
        root = AGE
        for att in nodes_atts[AGE - 1]:
            name = AGE_NAMES[att]
            nodeid += 1
            aux = anytree.Node(nodeid, display_name=name, is_class=False, parent=rootNode, s=[x for x in passengers if x[root] == att], att=att, nodes=[SEX, CLASS], entropies=age_entropies)
            surv = sum(1.0 if x[0] else 0.0 for x in aux.s)
            notSurv = sum(1.0 if not x[0] else 0.0 for x in aux.s)
            anytree.Node(-1, parent=aux, att=surv > notSurv)
        # nodes = [SEX, CLASS]
        # entropies = age_entropies
    else:
        root = SEX
        nodeid += 1
        rootNode = anytree.Node(nodeid, display_name="SEX", is_class=True)
        for att in nodes_atts[SEX - 1]:
            name = SEX_NAMES[att]
            nodeid += 1
            aux = anytree.Node(nodeid, display_name=name, is_class=False, parent=rootNode, s=[x for x in passengers if x[root] == att], att=att, nodes=[CLASS, AGE], entropies=sex_entropies)
            surv = sum(1.0 if x[0] else 0.0 for x in aux.s)
            notSurv = sum(1.0 if not x[0] else 0.0 for x in aux.s)
            anytree.Node(-1, parent=aux, att=surv > notSurv)
        # nodes = [CLASS, AGE]
        # entropies = sex_entropies

    ret = calc(rootNode, testing, passengers, False)
    testingPrec.append(ret[0])
    trainingPrec.append(ret[1])
    x_nodes.append(len(rootNode.descendants)+1)
    for n in rootNode.descendants:
        children = n.children
        n.children = [x for x in children if x.name != -1]

    stop = False
    while not stop:
        for treeNode in rootNode.leaves:
            maxGain = 0
            maxEntropies = {}
            node = -1
            for attribute in treeNode.nodes:
                s = treeNode.s
                if len(s) == 0:
                    continue
                # print("{} {}".format(treeNode, attribute))
                g = gain(s, nodes_atts[attribute - 1], attribute, treeNode.entropies[treeNode.att])
                if g[0] > maxGain:
                    maxGain = g[0]
                    maxEntropies = g[1]
                    node = attribute
            # print(i)
            # print(maxGain)
            # print(names[node])
            if node == -1:
                continue
            print("el nodo discriminante es " + names[node])
            parent = treeNode
            newNodes = [x for x in treeNode.nodes if x != node]
            if len(newNodes) == 0:
                stop = True
            nodeid += 1
            newNode = anytree.Node(nodeid, display_name=names[node], is_class=True, parent=parent, nodes=newNodes)
            for value in nodes_atts[node - 1]:
                name = nodes_names[node - 1][value]
                nodeid += 1
                aux = anytree.Node(nodeid, display_name=name, is_class=False, parent=newNode, nodes=newNodes, s=[x for x in treeNode.s if x[node] == value], att=value, entropies=maxEntropies)
            for n in rootNode.leaves:
                if not n.is_class and n.s is not None:
                    surv = sum(1.0 if x[0] else 0.0 for x in n.s)
                    notSurv = sum(1.0 if not x[0] else 0.0 for x in n.s)
                    anytree.Node(-1, parent=n, att=surv > notSurv)
            #check for draw
            # for pre, fill, node in anytree.RenderTree(rootNode):
            #     print("%s%s" % (pre, node.name))
            ret = calc(rootNode, testing, passengers, False)
            testingPrec.append(ret[0])
            trainingPrec.append(ret[1])
            x_nodes.append(len(rootNode.descendants) + 1)
            for n in rootNode.descendants:
                children = n.children
                n.children = [x for x in children if x.name != -1]

    parents = set()
    for node in rootNode.leaves:
        surv = sum(1.0 if x[0] else 0.0 for x in node.s)
        notSurv = sum(1.0 if not x[0] else 0.0 for x in node.s)
        nodeid += 1
        anytree.Node(nodeid, display_name="Survived" if surv > notSurv else "Died", parent=node, att=surv > notSurv, is_class=False)
        if node.parent.parent is not None:
            parents.add(node.parent.parent)

    for parent in parents:
        value = parent.leaves[0].att
        diff = False
        for leaf in parent.leaves:
            if value != leaf.att:
                diff = True
        if not diff:
            parent.children = []
            nodeid += 1
            anytree.Node(nodeid, display_name="Survived" if value else "Died", parent=parent, att=value, is_class=False)
    for pre, fill, node in anytree.RenderTree(rootNode):
        print("%s%s" % (pre, node.display_name))
    calc(rootNode, testing, passengers, True)

    #plot

    builds = np.array(x_nodes)
    y_stack = np.row_stack((trainingPrec, testingPrec))

    fig = plt.figure(figsize=(11, 8))
    ax1 = fig.add_subplot(111)

    ax1.plot(builds, y_stack[0, :], label='Training', color='c', marker='o')
    ax1.plot(builds, y_stack[1, :], label='Testing', color='g', marker='o')

    plt.xticks(builds)
    plt.xlabel('Nodes')

    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15, 1))
    # ax1.grid('on')
    ax1.grid()
    plt.show()
    # plt.savefig('smooth_plot.png')

    print(testingPrec)
    print(x_nodes)
    return [rootNode, passengers]
    # DotExporter(rootNode, nodeattrfunc=lambda node: 'label="{}"'.format(node.display_name)).to_picture("caca.png")


rootNodes = []
while len(rootNodes) < 10:
    rootNodes.append(run())

forestCalc(rootNodes)

