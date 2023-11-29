import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

    # 预处理SMD数据集
    df = pd.read_csv(PROJECT_ROOT + '/data/SMD.csv', index_col='Timestamp')

    df = df[:5500]

    df_norm = ((df - df.min()) / (df.max() - df.min())) * 50

    df_norm.to_csv(PROJECT_ROOT + './data/SMD_norm.csv')

    df_norm.plot(subplots=True, figsize=(10, 10))

    plt.show()
