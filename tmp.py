import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hypernetx
from sklearn.preprocessing import MinMaxScaler

import utils
from utils import project_root


PROJECT_ROOT = project_root()


if __name__ == '__main__':

    # 将csv转换成TANE可用的txt文件
    # df = pd.read_csv('./data/IDF.csv')
    # df.drop(['timestamp'], axis=1, inplace=True)
    #
    # for col in df.columns:
    #     df[col] = df[col] * 100
    #     df[col] = df[col].astype(int)
    #
    # df = df.iloc[:, :10]
    #
    # with open('idf_data.txt', 'w') as f:
    #     for line in df.values:
    #         for i in range(len(line)):
    #             x = line[i]
    #             f.write(str(x))
    #             if i < len(line) - 1:
    #                 f.write(',')
    #             else:
    #                 f.write('\n')

    # df = pd.read_csv(PROJECT_ROOT + '/PUMP.csv', index_col='time')
    # df.drop(columns=['Label'], inplace=True)
    #
    # # df = df[['sensor0', 'sensor1']]
    # df = df[30000: 50000]
    # df = df[['sensor0', 'sensor1', 'sensor2', 'sensor32', 'sensor33', 'sensor34']]
    #
    # for col in df.columns:
    #     values = df[col].values
    #
    #     speeds = np.diff(values)
    #     s_mean = speeds.mean()
    #     s_std = speeds.std()
    #
    #     s_lb = s_mean - 2 * s_std
    #     s_ub = s_mean + 2 * s_std
    #
    #     for i in range(10, len(values) - 10):
    #         if values[i] - values[i-1] > s_ub or values[i] - values[i-1] < s_lb:
    #             values[i] = np.mean(values[i-10:i+10])
    #
    # df.to_csv(PROJECT_ROOT + '/data/PUMP.csv')

    # df.plot(subplots=True, figsize=(10, 10))
    # plt.show()

    df = pd.read_csv(utils.project_root() + '/data/IDF.csv', index_col='timestamp')
    print(df)
    df.drop(columns=['U3_HNA10CP107'], inplace=True)
    df.to_csv(utils.project_root() + '/data/IDF.csv')

    pass
