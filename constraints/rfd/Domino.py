from relax import FD
from relax import RFD
import numpy as np
import math
import pandas as pd
import random
import sys
import os

'''
 判断是否全为1，用于集合覆盖过程，判断是否所有的数据都被覆盖
 @param S 集合，为一个01组成的数组
 @return bool True:全为1  False：不全为1
'''


def getDistanceRelation(r):
    res = []
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            kk = []
            for k in range(len(r[i])):
                kk.append(abs(r[i][k] - r[j][k]))
            res.append(kk)
    res = np.array(res)
    return res


def orderedRelation(diff_list, i):
    return diff_list[np.argsort(diff_list[:, i])]


def dominate(T_beta, diff_list_a):
    for h in T_beta:
        flag = True
        for j in range(len(h)):
            if h[j] > diff_list_a[j]:
                flag = False
        if flag:
            return True
    return False


def getTRel(diff_list_a, i):
    T = []
    T_beta = []
    w = len(diff_list_a)
    step = 1
    while w > 0:
        beta = max(w - step, 0)
        while beta != 0 and diff_list_a[beta - 1][i] == diff_list_a[beta][i]:
            beta = beta - 1
        for j in range(beta, w):
            if dominate(T_beta, diff_list_a[j]):
                continue
            T_beta.append(diff_list_a[j])
        T.append((T_beta, (i, diff_list_a[beta][i])))
        w = beta
        step = step * 2
    return T


def viable(X, DM):
    for h in DM:
        flag1 = False
        flag2 = True
        for f in X:
            if h[f] != 0:
                flag2 = False
            if h[f] < 0:
                flag1 = True
        if flag1 == False and flag2 == False:
            return False
    return True


def redundant(X_new, LHS):
    for i in LHS:
        d = [False for c in i not in X_new]
        if not d:
            return True
    return False


def generateRFDcs(T_beta, w, LHS):
    RFDcs = []
    for cc in LHS:
        LHS_now = []
        for i in cc:
            if i == cc[0]:
                LHS_now.append([i, w[i] - 0.001])
                continue
            LHS_now.append([i, w[i]])
        for j in range(1, len(cc)):
            kk = 99999999
            for i in T_beta:
                if i == w:
                    continue
                if T_beta[i][cc[j]] <= w[cc[j]]:
                    continue
                if T_beta[i][cc[0]] >= w[cc[0]]:
                    continue
                flag = True
                for e in range(j):
                    if T_beta[i][cc[e]] > LHS_now[e][1]:
                        flag = False
                if not flag:
                    continue
                flag = False
                for e in range(j + 1, len(cc)):
                    if T_beta[i][cc[e]] <= LHS_now[e][1]:
                        flag = True
                if not flag:
                    continue
                kk = min(kk, T_beta[i][cc[j]])
            if kk == 99999999:
                continue
            LHS_now[j][1] = kk - 0.001
        RFDcs.append(LHS_now)
    return RFDcs


def getRFDs(T_beta, A_i, w):
    L = 1
    levelAttr = []
    LHS = []
    DM = []
    for i in T_beta:
        if i.all() == w.all():
            continue
        kk = []
        for j in range(len(w)):
            kk.append(i[j] - w[j])
        DM.append(kk)
    for i in range(len(w)):
        if i == A_i:
            continue
        levelAttr.append([i])
    while L <= len(w) and len(levelAttr) != 0:
        L_nxt = []
        for X in levelAttr:
            if viable(X, DM):
                LHS.append(X)
        levelAttr = []
        for X in levelAttr:
            if not viable(X, DM):
                for kk in range(len(w)):
                    if kk == A_i:
                        continue
                    X_new = X
                    X_new.append(kk)
                    if not redundant(X_new, LHS):
                        levelAttr.append(X_new)
        L = L + 1
    return generateRFDcs(T_beta, w, LHS)


def my_print(ans, file):
    f = open(file, 'w')
    for i in ans:
        for j in i[0]:
            print("(%s , <= , %.2f)" % (name_list[j[0]], j[1]), end=' ', file=f)

        if len(i[0]) != 0:
            print(",", end='  ', file=f)
        print("(%s , <= , %.2f)" % (name_list[i[1][0]], i[1][1]), file=f)
    f.close()

def Domino(r):
    RFDcs = []
    diff_list = r
    cnt = 1
    for i in range(len(diff_list[0])):
        diff_list_a = orderedRelation(diff_list, i)
        T = getTRel(diff_list_a, i)
        for T_beta in T:
            RHS = T_beta[1]
            T_beta = T_beta[0]
            for w in T_beta:
                LHS_now = getRFDs(T_beta, i, w)
                for kk in LHS_now:
                    cnt += 1
                    if cnt & 10 == 10:
                        RFDcs.append((kk, RHS))
    return RFDcs


from pre import pre_glass


def test(Distance, name_list):
    import time

    start = time.time()
    ans = Domino(Distance)
    end = time.time()
    my_print(ans, "123.txt")
    print("No.1")
    print("Glass:", Distance.shape)
    print("Time:", (end - start))
    '''print("Score:", get_Score(Distance, ans))'''
    print("Cnt:", len(ans))


def make_data(Distance_new, rate):
    for i in range(len(Distance_new)):
        rad = random.random()
        if rad < rate:
            Distance_new[i] = Distance_new[i] * rate
    return Distance_new


def work():
    global name_list
    Distance, name_list = pre_glass(50, 6)
    test(Distance, name_list)


if __name__ == "__main__":
    work()
