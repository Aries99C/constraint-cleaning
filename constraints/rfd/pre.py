import numpy as np


def pre_glass(mts, n, m):
    data, cols = mts.clean.values, mts.cols

    data = np.array(data)
    data = data[:n, :m]
    Distance = getDistanceRelation(data)

    return Distance, cols


def getDistanceRelation(r):
    res = []
    for i in range(len(r) - 1):
        for j in range(i + 1, len(r)):
            kk = []
            for k in range(len(r[i])):
                kk.append(abs(r[i][k] - r[j][k]))
            res.append(kk)
    res = np.array(res)
    return res



