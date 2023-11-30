import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def mining_speed_constraints(mts, alpha=3, verbose=0):
    speed_constraints = {}  # 以键值对的形式返回

    speed_hist = {}  # 以键值对的形式存储每个属性的速度分布

    # 采用3-sigma法则进行挖掘
    for col in mts.clean.columns:  # 每列独立挖掘
        values = mts.clean[col].values  # 将列数组转换为ndarray
        speeds = np.diff(values)  # 差分计算速度
        speed_mean = np.mean(speeds)  # 速度的均值
        speed_std = np.std(speeds)  # 速度的标准差

        # 由3-sigma计算得到满足大部分数据的速度约束
        speed_lb = speed_mean - alpha * speed_std
        speed_ub = speed_mean + alpha * speed_std

        if speed_lb == 0. and speed_ub == 0.:
            speed_lb = -0.00001
            speed_ub = 0.00001

        # 存储当前列上的速度约束
        speed_constraints[col] = (speed_lb, speed_ub)
        if verbose == 2:  # 需要显示的情况才存储，节省内存
            speed_hist[col] = speeds

    if verbose > 0:  # 输出速度约束的形式
        print('{:=^80}'.format(' 在数据集{}上挖掘速度约束 '.format(mts.dataset.upper())))
        for item in speed_constraints.items():
            print('{:.2f} <= s({}) <= {:.2f}'.format(item[1][0], item[0], item[1][1]))
            if verbose == 2:  # 额外绘制速度直方图
                print(speed_hist[item[0]])
                plt.hist(speed_hist[item[0]])
                plt.show()

    return speed_constraints
