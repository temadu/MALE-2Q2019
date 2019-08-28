import csv

import numpy as np


prob=["","","",0]
dado=[1, True, "", ""]  # rank, gre, gpa, adm    "":= cualquier valor
adm = []
gre = []
gpa = []
rank = []
n_rank = [2, 2, 2, 2]
p_rank = [0, 0, 0, 0]
n_gpa_rank = [1, 1, 1, 1]  # R: 1,2,3,4
p_gpa_rank = [0, 0, 0, 0]  # R: 1,2,3,4
n_gre_rank = [1, 1, 1, 1]  # R: 1,2,3,4
p_gre_rank = [0, 0, 0, 0]  # R: 1,2,3,4
n_adm_gregpa = [1, 1, 1, 1]  # GrGp: TT, TF, FT, FF
p_adm_gregpa = [0, 0, 0, 0]  # GrGp: TT, TF, FT, FF
n_gregparank = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # RaGrGp: 1TT, 1TF, 1FT, 1FF, etc
n_adm_gregparank = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # RaGrGp: 1TT, 1TF, 1FT, 1FF, etc
p_adm_gregparank = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
possibilities = [[1, 2, 3, 4], [True, False], [True, False], [True, False]]
n = 400
with open('./data/admision.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    first = True
    # print(n)
    for row in reader:
        if first:
            first = False
            continue
        admElem = True if row[0] == "1" else False
        adm.append(admElem)
        greElem = True if float(row[1]) >= 500 else False
        gre.append(greElem)
        gpaElem = True if float(row[2]) >= 3 else False
        gpa.append(gpaElem)
        rankElem = int(row[3])
        rank.append(rankElem)
        n_rank[rankElem - 1] += 1
        p_rank[rankElem - 1] += 1./(n+4)

        if gpaElem and greElem:
            n_gregparank[4 * (rankElem - 1)] += 1
            n_adm_gregparank[4 * (rankElem - 1)] += admElem
            # n_adm_gregpa[0] += admElem
        elif not gpaElem and greElem:
            n_gregparank[4 * (rankElem - 1) + 1] += 1
            n_adm_gregparank[4 * (rankElem - 1) + 1] += admElem
            # n_adm_gregpa[1] += admElem
        elif gpaElem and not greElem:
            n_gregparank[4 * (rankElem - 1) + 2] += 1
            n_adm_gregparank[4 * (rankElem - 1) + 2] += admElem
            # n_adm_gregpa[2] += admElem
        else:
            n_gregparank[4 * (rankElem - 1) + 3] += 1
            n_adm_gregparank[4 * (rankElem - 1) + 3] += admElem
            # n_adm_gregpa[3] += admElem

        n_gpa_rank[rankElem - 1] += gpaElem
        n_gre_rank[rankElem - 1] += greElem
csvFile.close()

# print(p_rank)
#
# print(n_gpa_rank)
# print(n_gre_rank)
# print(n_rank)
p_gpa_rank = np.array(n_gpa_rank)/np.array(n_rank)
p_gre_rank = np.array(n_gre_rank)/np.array(n_rank)
# print()
# print(p_gpa_rank)
# print(p_gre_rank)
# print()
# print(n_adm_gregparank)
# print(n_gregparank)
# print()

p_adm_gregparank = np.array(n_adm_gregparank)/np.array(n_gregparank)
# print(p_adm_gregparank)

# ya tengo todas las probabilidades que necesitaria


# para el 1
# arriba necesito la suma de p(1,f,f,F) + p(1,t,f,F) + p(1,f,t,F) + p(1,t,t,F) aka
# dividendo necesito la suma de p(1,f,f,f) + p(1,t,f,f) + p(1,f,t,f) + p(1,t,t,f) + p(1,f,f,t) + p(1,t,f,t) + p(1,f,t,t) + p(1,t,t,t)
# necesito la suma de varias p sobre la suma de varias p


# RaGrGp: 1TT, 1TF, 1FT, 1FF, etc

def p(rank, gre, gpa, adm):
    return abs(1*(not adm) - p_adm_gregparank[(rank-1) * 4 + (not gre) * 2 + (not gpa)])


# prob=["","","",0]
# dado=[1, "", "", ""]  # rank, gre, gpa, adm    "":= cualquier valor

comb=[-1,-1,-1,-1]
for n in range(0,4):
    if dado[n] != "":
        comb[n] = dado[n]
    else:
        comb[n] = prob[n]
# print(comb)

# tengo 32 combinaciones
# si 0 es "" y el resto no tnego 4 combinaciones    OK
# si 1 es "" y el resto no tengo 2 combinaciones    OK
# si 2 es "" y el resto no tengo 2 combinaciones
# si 3 es "" y el resto no tengo 2 combinaciones
# si 0y1 es "" y el resto no tengo 8 combinaciones   OK
# si 0y2 es "" y el resto no tengo 8 combinaciones    OK
# si 0y3 es "" y el resto no tengo 8 combinaciones    OK
# si 1y2 es "" y el resto no tengo 4 combinaciones    OK
# si 1y3 es "" y el resto no tengo 4 combinaciones   OK
# si 2y3 es "" y el resto no tengo 4 combinaciones
# si 0y1y2 es "" y el resto no tengo 16 combinaciones  OK
# si 0y1y3 es "" y el resto no tengo 16 combinaciones  OK
# si 0y2y3 es "" y el resto no tengo 16 combinaciones  OK
# si 1y2y3 es "" y el resto no tengo 8 combinaciones   OK
# si ninguno es "" tengo 1 combinacion
coco1285 = 0


def superFunction(comb):
    coco = 0
    if comb[0] == "":
        if comb[1] == "":
            if comb[2] == "":
                for i in range(0,2):
                    for j in range(0,2):
                        for k in range(0,4):
                            coco += p(possibilities[0][k], possibilities[1][j], possibilities[2][i], comb[3])
            elif comb[3] == "":
                for i in range(0,2):
                    for j in range(0,2):
                        for k in range(0,4):
                            coco+= p(possibilities[0][k], possibilities[1][j], comb[2], possibilities[3][i])
            else:
                for j in range(0, 2):
                    for k in range(0, 4):
                        coco += p(possibilities[0][k], possibilities[1][j], comb[2], comb[3])
        elif comb[2] == "":
            if comb[3] == "":
                for i in range(0,2):
                    for j in range(0,2):
                        for k in range(0,4):
                            coco+= p(possibilities[0][k], comb[1], possibilities[2][j], possibilities[3][i])
            else:
                for j in range(0, 2):
                    for k in range(0, 4):
                        coco += p(possibilities[0][k], comb[1], possibilities[2][j], comb[3])
        elif comb[3] == "":
            for j in range(0, 2):
                for k in range(0, 4):
                    coco += p(possibilities[0][k], comb[1], comb[2], possibilities[3][j])
        else:
            for k in range(0, 4):
                coco += p(possibilities[0][k], comb[1], comb[2], comb[3])
    elif comb[1] == "":
        if comb[2] == "":
            if comb[3] == "":
                for i in range(0,2):
                    for j in range(0,2):
                        for k in range(0,2):
                            coco+= p(comb[0], possibilities[1][k], possibilities[2][j], possibilities[3][i])
            else:
                for i in range(0,2):
                    for j in range(0,2):
                        coco+= p(comb[0], possibilities[1][j],  possibilities[2][i], comb[3])
        elif comb[3] == "":
            for i in range(0, 2):
                for j in range(0, 2):
                    coco += p(comb[0], possibilities[1][j], comb[2], possibilities[3][i])
        else:
            for i in range(0, 2):
                coco += p(comb[0], possibilities[1][i], comb[2], comb[3])
    elif comb[2] == "":
        if comb[3] == "":
            for i in range(0, 2):
                for j in range(0, 2):
                    coco += p(comb[0], comb[1], possibilities[2][j], possibilities[3][i])
        else:
            for i in range(0, 2):
                coco += p(comb[0], comb[1], possibilities[2][i], comb[3])
    elif comb[3] == "":
        for i in range(0, 2):
            coco += p(comb[0], comb[1], comb[2], possibilities[3][i])
    else:
        coco += p(comb[0], comb[1], comb[2], comb[3])
    return coco


print('P({} {} {} {} | {} {} {} {}) = {}'.format(
    prob[0] if prob[0] == "" else "rank="+str(prob[0]),
    prob[1] if prob[1] == "" else "GRE="+str(prob[1]),
    prob[2] if prob[2] == "" else "GPA="+str(prob[2]),
    prob[3] if prob[3] == "" else "adm="+str(prob[3]),
    dado[0] if dado[0] == "" else "rank="+str(dado[0]),
    dado[1] if dado[1] == "" else "GRE="+str(dado[1]),
    dado[2] if dado[2] == "" else "GPA="+str(dado[2]),
    dado[3] if dado[3] == "" else "adm="+str(dado[3]),
    superFunction(comb)/superFunction(dado)))
# print("------")

# print(superFunction(comb))
# print(superFunction(dado))
# print(superFunction(comb)/superFunction(dado))
