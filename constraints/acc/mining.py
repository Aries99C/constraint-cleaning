import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def mining_acc_constraints(mts, alpha=3, verbose=0):
    acc_constraints = {}    # 以键值对的形式返回

    acc_hist = {}   # 以键值对的形式存储每个属性的速度分布

    # 采用3-sigma法则进行挖掘
    for col in mts.clean.columns:   # 每列独立挖掘
        values = mts.clean[col].values  # 将列数组转换为ndarray
        speeds = np.diff(values)    # 差分计算速度
        accs = np.diff(speeds)      # 差分计算加速度
        acc_mean = np.mean(accs)    # 加速度均值
        acc_std = np.std(accs)      # 加速度标准差

        # 由3-sigma计算得到满足大部分数据的速度约束
        acc_lb = acc_mean - alpha * acc_std
        acc_ub = acc_mean + alpha * acc_std

        # 存储当前列上的加速度约束
        acc_constraints[col] = (acc_lb, acc_ub)
        if verbose == 2:    # 需要显示的情况下才存储
            acc_hist[col] = accs

    if verbose > 0:  # 输出加速度约束的形式
        print('{:=^80}'.format(' 在数据集{}上挖掘加速度约束 '.format(mts.dataset.upper())))
        for item in acc_constraints.items():
            print('{:.2f} <= a({}) <= {:.2f}'.format(item[1][0], item[0], item[1][1]))
            if verbose == 2:  # 额外绘制速度直方图
                print(acc_hist[item[0]])
                plt.hist(acc_hist[item[0]])
                plt.show()

