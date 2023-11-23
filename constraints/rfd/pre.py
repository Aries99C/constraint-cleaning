import numpy as np
import os
import pandas as pd


def read_data(file_name):
    path = os.getcwd() + file_name
    f = open(path, encoding='utf-8')
    data = pd.read_csv(f)
    name_list = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides"
        , "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality", "Id"]
    data_x = data[name_list]
    x = data_x.values
    return x, name_list


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


def pre_glass(n, m):
    data, label = read_data('\\WineQT.csv')

    data = np.array(data)
    data = data[:n, :m]
    Distance = getDistanceRelation(data)

    return Distance, label
