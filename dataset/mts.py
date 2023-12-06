import random
import math

import hypernetx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from utils import project_root
from dataset import DATASETS
from constraints.speed.mining import mining_speed_constraints
from constraints.acc.mining import mining_acc_constraints
from constraints.stcd.mining import mining_stcd
from constraints.fd.mining import read_from_TANE
from constraints.rfd.Domino import domino_mining_rfd
from constraints.rfd.CORDS import cords_mining_rfd
from constraints.rfd.IsCover import is_cover_mining_rfd

from cleaning.benchmark import delta, raa, check_repair_violation, violation_rate, speed_local, speed_global, acc_local, acc_global, IMR, ewma, median_filter, func_lp, func_mvc
from cleaning.benchmark import f1, fd_detect
from cleaning.benchmark import array2window

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 2000)

PROJECT_ROOT = project_root()


class MTS(object):
    """
    多元时序数据集类，用于存储与数据集相关的所有信息，支持以下功能：
    1. 支持DataFrame类型与其他核心数据结构的转化，包括：HoloClean中Dataset类、LearnCRR中的DataBase类、SCREEN和IMR中的TimeSeries类
    2. 挖掘并存储数据集中的置信度高的约束，包括：否定约束、速度约束、加速度约束、方差约束、行列约束
    """

    def __init__(self, dataset='idf', index_col='timestamp', datetime_index=True, size=5000, rfd_m=10, verbose=0):
        """
        初始化多元时序数据集，读取数据集并以最通用的DataFrame类型存储
        :param dataset: 数据集名称标识
        :param index_col: 数据集中作为时间戳索引的列名称
        :param datetime_index: 标识数据集中的时间戳索引是否为datetime类型，如果是[1,2,...]的形式请将该参数设置为False
        :param size: 从数据集中截取的长度
        :param verbose: 日志显示，0为不输出日志，1输出基本的数据集信息，2额外输出数据集概览
        """
        self.dataset = None             # 数据集标识
        self.len = None                 # 数据集长度
        self.dim = None                 # 数据集维数
        self.cols = None                # 数据集属性集合

        self.origin = None              # 观测值
        self.clean = None               # 正确值
        self.modified = None            # 修复值

        self.isLabel = None             # IMR用标签
        self.isModified = None          # 记录修复单元格
        self.isDirty = None             # 记录注入错误的单元格

        self.mining_constraints = None  # 使用的约束方法
        self.speed_constraints = None   # 速度约束
        self.acc_constraints = None     # 加速度约束
        self.stcds = None               # 时窗约束
        self.fds = None                 # 函数依赖
        self.domino = None              # Domino挖掘RFD
        self.cords = None               # CORDS挖掘RFD
        self.is_cover = None            # IsCover挖掘RFD

        self.rfd_m = rfd_m              # RFD挖掘和检测时使用的列数

        assert dataset in DATASETS.keys()   # 保证使用了预设的数据集，请在__init__.py文件中配置

        self.dataset = dataset
        self.clean = pd.read_csv(                                           # 默认读取干净数据
            PROJECT_ROOT + DATASETS.get(dataset),                           # 绝对路径+配置路径
            sep=',',                                                        # 默认csv文件用','间隔
        )
        if datetime_index:  # 默认索引列为时间戳
            self.clean[index_col] = pd.to_datetime(self.clean[index_col])   # 首先将索引列转化为datetime64类型
        self.clean.set_index(index_col, inplace=True)   # 设置索引列

        assert size <= len(self.clean)              # 保证预设读取的长度≤数据集总长度
        self.len = size                             # 获取数据集长度
        self.dim = len(self.clean.columns)          # 获取数据集维数
        self.cols = self.clean.columns.tolist()     # 获取数据集属性集合

        self.clean = self.clean[0: self.len]        # 截取预设长度的数据

        if verbose > 0:         # 日志显示，2级比1级更详细
            print('{:=^80}'.format(' 加载数据集{} '.format(self.dataset.upper())))
            print('数据集长度: ' + str(self.len))
            print('数据集维数: ' + str(self.dim))
            if verbose == 2:    # 显示数据集概览
                print(self.clean)

    def constraints_mining(self, pre_mined=False, mining_constraints=None, w=2, n_component=1, confidence=0.999, verbose=0):
        """
        根据多元时序数据的正确值挖掘规则，包括行约束和列约束
        :param pre_mined: 预挖掘标记，为True时直接读取已挖掘的约束
        :param mining_constraints: 计划使用的约束类型
        :param verbose: 日志显示
        """
        if mining_constraints is None:
            self.mining_constraints = ['speed', 'acc', 'stcd']  # 默认挖掘的约束
        else:
            self.mining_constraints = mining_constraints        # 自定义约束
        if pre_mined:   # 使用预先挖掘好的规则，直接读取即可
            if 'speed' in self.mining_constraints:  # 支持速度约束
                with open(PROJECT_ROOT + '/constraints/rules/{}_speed.txt'.format(self.dataset), 'rb') as f:
                    self.speed_constraints = pickle.load(f)
                    print('{:=^80}'.format(' 读取数据集{}上的速度约束 '.format(self.dataset.upper())))
                    print('约束数量: {}'.format(self.dim))
                    if verbose > 0:  # 输出速度约束的形式
                        for item in self.speed_constraints.items():
                            print('{:.2f} <= s({}) <= {:.2f}'.format(item[1][0], item[0], item[1][1]))
            if 'acc' in self.mining_constraints:    # 支持加速度约束
                with open(PROJECT_ROOT + '/constraints/rules/{}_acc.txt'.format(self.dataset), 'rb') as f:
                    self.acc_constraints = pickle.load(f)
                    print('{:=^80}'.format(' 读取数据集{}上的加速度约束 '.format(self.dataset.upper())))
                    print('约束数量: {}'.format(self.dim))
                    if verbose > 0:  # 输出加速度约束的形式
                        for item in self.acc_constraints.items():
                            print('{:.2f} <= s({}) <= {:.2f}'.format(item[1][0], item[0], item[1][1]))
            if 'stcd' in self.mining_constraints:   # 支持时窗约束
                with open(PROJECT_ROOT + '/constraints/rules/{}_stcd.txt'.format(self.dataset), 'rb') as f:
                    self.stcds = pickle.load(f)
                    print('{:=^80}'.format(' 读取数据集{}上的时窗线性约束 '.format(self.dataset.upper())))
                    print('约束数量: {}'.format(len(self.stcds)))
                    if verbose > 0:  # 输出时窗约束的形式
                        for rule in self.stcds:
                            print(rule)
            if 'crr' in self.mining_constraints:        # 支持crr
                pass
            if 'domino' in self.mining_constraints:     # 支持rfd
                with open(PROJECT_ROOT + '/constraints/rules/{}_rfd_domino.txt'.format(self.dataset), 'rb') as f:
                    self.domino = pickle.load(f)
                    print('{:=^80}'.format(' 读取数据集{}上的松弛函数依赖-Domino '.format(self.dataset.upper())))
                    print('约束数量: {}'.format(len(self.domino)))
                    if verbose > 0: # 输出松弛函数依赖
                        for conditions, y in self.domino:
                            if len(conditions) == 0:
                                print('松弛函数依赖: d({}) <= {:.3f}'.format(y[0], y[1]))
                            else:
                                print('松弛函数依赖: {} --> {}'.format(
                                    ['d({}) <= {:.3f}'.format(cond[0], cond[1]) for cond in conditions],
                                    'd({}) <= {:.3f}'.format(y[0], y[1])))
            if 'cords' in self.mining_constraints:     # 支持rfd
                with open(PROJECT_ROOT + '/constraints/rules/{}_rfd_cords.txt'.format(self.dataset), 'rb') as f:
                    self.cords = pickle.load(f)
                    print('{:=^80}'.format(' 读取数据集{}上的松弛函数依赖-CORDS '.format(self.dataset.upper())))
                    print('约束数量: {}'.format(len(self.cords)))
                    if verbose > 0: # 输出松弛函数依赖
                        for conditions, y in self.cords:
                            if len(conditions) == 0:
                                print('松弛函数依赖: d({}) <= {:.3f}'.format(y[0], y[1]))
                            else:
                                print('松弛函数依赖: {} --> {}'.format(
                                    ['d({}) <= {:.3f}'.format(cond[0], cond[1]) for cond in conditions],
                                    'd({}) <= {:.3f}'.format(y[0], y[1])))
            if 'is_cover' in self.mining_constraints:     # 支持rfd
                with open(PROJECT_ROOT + '/constraints/rules/{}_rfd_is_cover.txt'.format(self.dataset), 'rb') as f:
                    self.is_cover = pickle.load(f)
                    print('{:=^80}'.format(' 读取数据集{}上的松弛函数依赖-IsCover '.format(self.dataset.upper())))
                    print('约束数量: {}'.format(len(self.is_cover)))
                    if verbose > 0: # 输出松弛函数依赖
                        for conditions, y in self.is_cover:
                            if len(conditions) == 0:
                                print('松弛函数依赖: d({}) <= {:.3f}'.format(y[0], y[1]))
                            else:
                                print('松弛函数依赖: {} --> {}'.format(
                                    ['d({}) <= {:.3f}'.format(cond[0], cond[1]) for cond in conditions],
                                    'd({}) <= {:.3f}'.format(y[0], y[1])))
            if 'fd' in self.mining_constraints:         # 支持fd
                with open(PROJECT_ROOT + '/constraints/rules/{}_fd.txt'.format(self.dataset), 'rb') as f:
                    self.fds = pickle.load(f)
                    print('{:=^80}'.format(' 读取数据集{}上的函数依赖 '.format(self.dataset.upper())))
                    print('约束数量: {}'.format(len(self.fds)))
                    if verbose > 0:  # 输出函数依赖的形式
                        for lhs, rhs in self.fds:
                            print('函数依赖: {} -> {}'.format([var[0] for var in lhs], [var[0] for var in rhs]))
        else:           # 否则需要挖掘规则
            if 'speed' in self.mining_constraints:  # 支持速度约束
                self.speed_constraints = mining_speed_constraints(self, alpha=3, verbose=verbose)
                with open(PROJECT_ROOT + '/constraints/rules/{}_speed.txt'.format(self.dataset), 'wb') as f:
                    pickle.dump(self.speed_constraints, f)  # pickle序列化速度约束
            if 'acc' in self.mining_constraints:    # 支持加速度约束
                self.acc_constraints = mining_acc_constraints(self, alpha=3, verbose=verbose)
                with open(PROJECT_ROOT + '/constraints/rules/{}_acc.txt'.format(self.dataset), 'wb') as f:
                    pickle.dump(self.acc_constraints, f)  # pickle序列化加速度约束
            if 'stcd' in self.mining_constraints:   # 支持时窗约束
                self.stcds = mining_stcd(self, win_size=w, n_components=n_component, confidence=confidence, verbose=verbose)
                with open(PROJECT_ROOT + '/constraints/rules/{}_stcd.txt'.format(self.dataset), 'wb') as f:
                    pickle.dump(self.stcds, f)  # pickle序列化时窗约束
            if 'crr' in self.mining_constraints:    # 支持crr
                pass
            if 'domino' in self.mining_constraints:    # 支持rfd
                self.domino = domino_mining_rfd(self, n=self.len, m=self.rfd_m, verbose=verbose)
                with open(PROJECT_ROOT + '/constraints/rules/{}_rfd_domino.txt'.format(self.dataset), 'wb') as f:
                    pickle.dump(self.domino, f)  # pickle序列化函数依赖
            if 'cords' in self.mining_constraints:     # 支持rfd
                self.cords = cords_mining_rfd(self, n=self.len, m=self.rfd_m, verbose=verbose)
                with open(PROJECT_ROOT + '/constraints/rules/{}_rfd_cords.txt'.format(self.dataset), 'wb') as f:
                    pickle.dump(self.cords, f)  # pickle序列化函数依赖
            if 'is_cover' in self.mining_constraints:     # 支持rfd
                self.is_cover = is_cover_mining_rfd(self, n=self.len, m=self.rfd_m, verbose=verbose)
                with open(PROJECT_ROOT + '/constraints/rules/{}_rfd_is_cover.txt'.format(self.dataset), 'wb') as f:
                    pickle.dump(self.is_cover, f)  # pickle序列化函数依赖
            if 'fd' in self.mining_constraints:     # 支持fd
                self.fds = read_from_TANE(self, verbose=verbose)
                with open(PROJECT_ROOT + '/constraints/rules/{}_fd.txt'.format(self.dataset), 'wb') as f:
                    pickle.dump(self.fds, f)  # pickle序列化函数依赖

    def clean2array(self):
        d = self.clean.copy(deep=True)      # 拷贝正确值
        return d.values                     # 将Dataframe转换为ndarray类型

    def modified2array(self):
        d = self.modified.copy(deep=True)   # 拷贝值
        return d.values                     # 将Dataframe转换为ndarray类型

    def insert_error(self, ratio=0.1, label_ratio=0.1, snr=10, verbose=0):
        # 拷贝正确值作为观测值
        self.origin = self.clean.copy(deep=True)

        # 初始化记录错误单元格的矩阵，False代表单元格不存在错误
        self.isDirty = self.origin.copy(deep=True)
        for col in self.isDirty.columns:
            self.isDirty[col] = False       # 全部初始化为False

        label_cnt = 0
        # 初始化IMR算法所使用的标签
        self.isLabel = self.origin.copy(deep=True)
        for col in self.isLabel.columns:
            self.isLabel[col] = False       # 全部初始化为False
            while self.isLabel[col].values.tolist().count(True) < int(self.len * label_ratio):  # 以固定比例随机生成标签
                random_label_pos = random.randint(0, self.len - 1)
                if not self.isLabel[col].values[random_label_pos]:
                    self.isLabel[col].values[random_label_pos] = True
                    label_cnt += 1
        self.isLabel[:5] = True             # 为IMR算法提供最基础的标签

        # 向观测值注入错误
        # 连续错误 U3_HNC10CT111
        self.origin['U3_HNC10CT111'].values[30: 45] += 2.
        self.isDirty['U3_HNC10CT111'].values[30: 45] = True
        self.isLabel['U3_HNC10CT111'].values[36] = True
        # 趋势错误
        self.origin['U3_HNC10CT111'].values[80: 95] += np.array([x * 0.1 for x in range(15)])
        self.isDirty['U3_HNC10CT111'].values[80: 95] = True
        self.isLabel['U3_HNC10CT111'].values[90] = True
        # 反向趋势错误
        self.origin['U3_HNC10CT111'].values[200: 215] += np.array([x * 0.1 for x in range(15, 0, -1)])
        self.isDirty['U3_HNC10CT111'].values[200: 215] = True
        self.isLabel['U3_HNC10CT111'].values[210] = True

        # 先根据信噪比生成白噪声
        noise_df = self.clean.copy(deep=True)   # 拷贝DataFrame格式
        for attr in noise_df.columns:
            x = self.origin[attr].values                        # 获取待注入信号
            noise = np.random.randn(self.len)                   # 生成正态分布随机数
            x_power = np.sum(x * x) / self.len                  # 计算信号平均能量
            noise_power = np.sum(noise * noise) / self.len      # 计算噪声平均能量
            noise_var = x_power / (math.pow(10., (snr / 10)))   # 根据设定的信噪比计算噪声方差，默认信噪比为10dB
            noise = math.sqrt(noise_var / noise_power) * noise  # 生成白噪声
            noise_df[attr] = noise                              # 白噪声替换对应列

        insert_size = 0
        error_size = int(ratio * self.len * self.dim)   # 计算错误总单元格
        while error_size > 0:
            insert_attrs = random.sample(self.cols, max(int(ratio * self.dim), 2))      # 随机注入部分属性
            insert_len = random.randint(10, 50)                             # 随机的错误长度
            insert_pos = random.randint(50, self.len - insert_len - 1)      # 随机的注入位置
            for attr in insert_attrs:
                if self.isDirty[attr].values[insert_pos]:   # 若已经注入过错误就跳过
                    continue
                self.origin[attr].values[insert_pos: insert_pos + insert_len] \
                    += noise_df[attr].values[insert_pos: insert_pos + insert_len]       # 注入白噪声
                error_size -= insert_len                                                # 更新错误数量
                insert_size += insert_len                                               # 记录注入错误数量
                self.isDirty[attr].values[insert_pos: insert_pos + insert_len] = True   # 更新记录错误的单元格

        # 拷贝待修复值，后续将在待修复值上进行修改
        self.modified = self.origin.copy(deep=True)

        self.isModified = self.origin.copy(deep=True)
        for col in self.isModified.columns:
            self.isModified[col] = False    # 全部初始化为False

        error_aver = self.delta_clean_origin()  # 平均误差

        print('{:=^80}'.format(' 向数据集{}注入错误 '.format(self.dataset.upper())))
        print('共计注入错误单元格数: {}'.format(insert_size))
        print('共计提供标签数：{}'.format(label_cnt))
        if verbose > 0:     # 日志显示
            # print('正确值和观测值的总误差: {:.3f}'.format(error_sum))
            print('注入错误的平均误差: {:.3f}'.format(error_aver))
            if verbose == 2:
                self.modified.plot(subplots=True, figsize=(20, 40))
                plt.show()

        return insert_size, error_aver

    def delta_clean_origin(self):
        """
        计算观测值和正确值的距离
        """
        df = self.clean.sub(self.origin)    # 两个mts相减
        x = np.absolute(df.values)          # 获取ndarray的绝对值
        return np.sum(x) / (df.shape[0] * df.shape[1])


if __name__ == '__main__':
    # 实验参数
    w = 2

    idf = MTS('idf', 'timestamp', True, size=20000, verbose=1)                     # 读取数据集
    # idf = MTS('SWaT', 'Timestamp', False, size=2000, verbose=1)
    # idf = MTS('WADI', 'Row', False, size=2000, verbose=1)
    # idf = MTS('PUMP', 'time', False, size=5000, verbose=1)

    idf.constraints_mining(w=w, confidence=0.95, verbose=1)                        # 挖掘约束
    # idf.constraints_mining(pre_mined=True, verbose=1)                               # 预配置约束集合
    # idf.insert_error(snr=15, verbose=1)                                             # 注入噪声

    # 为HoloClean方法生成数据集和规则
    # idf.origin = idf.origin.round(2)
    # idf.origin.to_csv(PROJECT_ROOT + '/mts.csv', index=None)
    #
    # with open(PROJECT_ROOT + '/mts_speed_constraint.txt', 'w') as f:
    #     for constraint in idf.speed_constraints.items():
    #         attr = constraint[0]
    #         lb, ub = constraint[1]
    #         f.write('{},{},{}\n'.format(attr, lb, ub))
    #
    # idf.clean = idf.clean.round(2)
    # clean = pd.DataFrame(columns=['tid', 'attribute', 'correct_val'])
    # for i in range(idf.len):
    #     for col in idf.clean.columns:
    #         if idf.isDirty[col].values[i]:
    #             clean.loc[len(clean)] = [i, col, idf.clean[col].values[i]]
    # clean.to_csv(PROJECT_ROOT + '/mts_clean.csv', index=None)
    # idf.clean.to_csv(PROJECT_ROOT + '/mts_label.csv', index=None)
    #
    # with open(PROJECT_ROOT + '/mts_stcd.txt', 'w') as f:
    #     for stcd in idf.stcds:
    #         for i in range(len(stcd.alpha)):
    #             f.write('{}'.format(stcd.alpha[i]))
    #             if i < len(stcd.alpha) - 1:
    #                 f.write(',')
    #             else:
    #                 f.write(';')
    #         f.write('{};'.format(stcd.func['intercept']))
    #         f.write('{},{}\n'.format(stcd.lb, stcd.ub))

    # 修复
    # # 速度约束Local修复
    # speed_local_modified, speed_local_is_modified, speed_local_time = speed_local(idf, w=w)
    # print('{:=^80}'.format(' 局部速度约束修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(speed_local_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(speed_local_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, speed_local_modified)))
    # print('修复后约束违反程度: {:.4g}'.format(check_repair_violation(speed_local_modified, idf.stcds, w)))
    # print('修复后约束违反率: {:.4g}'.format(violation_rate(speed_local_modified, idf.stcds, w)))
    #
    # # 速度约束Global修复
    # speed_global_modified, speed_global_is_modified, speed_global_time = speed_global(idf, w=w, x=10)
    # print('{:=^80}'.format(' 全局速度约束修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(speed_global_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(speed_global_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, speed_global_modified)))
    # print('修复后约束违反程度: {:.4g}'.format(check_repair_violation(speed_global_modified, idf.stcds, w)))
    # print('修复后约束违反率: {:.4g}'.format(violation_rate(speed_global_modified, idf.stcds, w)))
    #
    # # 速度约束+加速度约束Local修复
    # acc_local_modified, acc_local_is_modified, acc_local_time = acc_local(idf)
    # print('{:=^80}'.format(' 局部速度约束+加速度约束修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(acc_local_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(acc_local_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, acc_local_modified)))
    # print('修复后约束违反程度: {:.4g}'.format(check_repair_violation(acc_local_modified, idf.stcds, w)))
    # print('修复后约束违反率: {:.4g}'.format(violation_rate(acc_local_modified, idf.stcds, w)))
    #
    # 速度约束+加速度约束Global修复
    # acc_global_modified, acc_global_is_modified, acc_global_time = acc_global(idf)
    # print('{:=^80}'.format(' 全局速度约束+加速度约束修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(acc_global_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(acc_global_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, acc_global_modified)))
    # print('修复后约束违反程度: {:.4g}'.format(check_repair_violation(acc_global_modified, idf.stcds, w)))
    # print('修复后约束违反率: {:.4g}'.format(violation_rate(acc_global_modified, idf.stcds, w)))
    #
    # # IMR修复
    # imr_modified, imr_is_modified, imr_time = IMR(max_iter=1000).clean(idf)
    # print('{:=^80}'.format(' IMR算法修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(imr_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(imr_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, imr_modified)))
    # print('修复后约束违反程度: {:.4g}'.format(check_repair_violation(imr_modified, idf.stcds, w)))
    # print('修复后约束违反率: {:.4g}'.format(violation_rate(imr_modified, idf.stcds, w)))
    #
    # # EWMA修复
    # ewma_modified, ewma_is_modified, ewma_time = ewma(idf, beta=0.9)
    # print('{:=^80}'.format(' EWMA算法修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(ewma_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(ewma_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, ewma_modified)))
    # print('修复后约束违反程度: {:.4g}'.format(check_repair_violation(ewma_modified, idf.stcds, w)))
    # print('修复后约束违反率: {:.4g}'.format(violation_rate(ewma_modified, idf.stcds, w)))
    #
    # # 中值滤波器修复
    # median_filter_modified, median_filter_is_modified, median_filter_time = median_filter(idf, w=10)
    # print('{:=^80}'.format(' 中值滤波修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(median_filter_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(median_filter_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, median_filter_modified)))
    # print('修复后约束违反程度: {:.4g}'.format(check_repair_violation(median_filter_modified, idf.stcds, w)))
    # print('修复后约束违反率: {:.4g}'.format(violation_rate(median_filter_modified, idf.stcds, w)))
    #
    # # func-LP修复
    # func_lp_modified, func_lp_is_modified, func_lp_time = func_lp(idf, w=w)
    # print('{:=^80}'.format(' func-LP修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(func_lp_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(func_lp_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, func_lp_modified)))
    # print('修复后约束违反程度: {:.4g}'.format(check_repair_violation(func_lp_modified, idf.stcds, w)))
    # print('修复后约束违反率: {:.4g}'.format(violation_rate(func_lp_modified, idf.stcds, w)))
    #
    # # func-mvc修复sorted
    # func_mvc_modified, func_mvc_is_modified, func_mvc_time = func_mvc(idf, w=w, mvc='sorted')
    # print('{:=^80}'.format(' func-MVC修复数据集{} '.format(idf.dataset.up per())))
    # print('修复用时: {:.4g}ms'.format(func_mvc_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(func_mvc_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, func_mvc_modified)))
    # print('修复后约束违反程度: {:.4g}'.format(check_repair_violation(func_mvc_modified, idf.stcds, w)))
    # print('修复后约束违反率: {:.4g}'.format(violation_rate(func_mvc_modified, idf.stcds, w)))

    # idf.clean.plot(subplots=True, figsize=(10, 10))
    # idf.origin.plot(subplots=True, figsize=(10, 10))
    # func_lp_modified.plot(subplots=True, figsize=(10, 10))
    # func_mvc_modified.plot(subplots=True, figsize=(10, 10))
    # plt.show()

    # fd_modified, fd_is_modified, fd_time = fd_detect(idf)
    # fd_p, fd_r, fd_f1 = f1(fd_is_modified, idf.isDirty)
    # print('{:=^80}'.format(' 函数依赖FD检测数据集{} '.format(idf.dataset.upper())))
    # print('检测用时: {:.4g}ms'.format(fd_time))
    # print('Precision: {:.4g}, Recall: {:.4g}, F1: {:.4g}'.format(fd_p, fd_r, fd_f1))

    # 修复示例图
    # speed_local_modified, speed_local_is_modified, speed_local_time = speed_local(idf, w=w)
    # speed_global_modified, speed_global_is_modified, speed_global_time = speed_global(idf, w=w, x=10)
    # acc_local_modified, acc_local_is_modified, acc_local_time = acc_local(idf)
    # acc_global_modified, acc_global_is_modified, acc_global_time = acc_global(idf)
    # imr_modified, imr_is_modified, imr_time = IMR(max_iter=1000).clean(idf)
    # ewma_modified, ewma_is_modified, ewma_time = ewma(idf, beta=0.9)
    # median_filter_modified, median_filter_is_modified, median_filter_time = median_filter(idf, w=10)
    # func_lp_modified, func_lp_is_modified, func_lp_time = func_lp(idf, w=w)
    # func_mvc_modified, func_mvc_is_modified, func_mvc_time = func_mvc(idf, w=w, mvc='sorted')

    # start = -1
    # end = -1
    # for i in range(300, idf.len):
    #     if start == -1 and idf.isDirty['U3_HNC10CT111'].values[i]:
    #         start = i
    #     if (not start == -1) and (not idf.isDirty['U3_HNC10CT111'].values[i]):
    #         end = i
    #         break

    # df = pd.DataFrame(columns=['tid', 'clean', 'origin',
    #                            'LP', 'approx',
    #                            'speed(L)', 'speed(G)', 'Acc(L)', 'Acc(G)',
    #                            'IMR', 'Median', 'EWMA'])
    # idx = 0
    # # for i in range(start - 5, end + 5):
    # for i in range(200 - 5, 215 + 5):
    #     values = [i, idf.clean['U3_HNC10CT111'].values[i], idf.origin['U3_HNC10CT111'].values[i],
    #               func_lp_modified['U3_HNC10CT111'].values[i], func_mvc_modified['U3_HNC10CT111'].values[i],
    #               speed_local_modified['U3_HNC10CT111'].values[i], speed_global_modified['U3_HNC10CT111'].values[i],
    #               acc_local_modified['U3_HNC10CT111'].values[i], acc_global_modified['U3_HNC10CT111'].values[i],
    #               imr_modified['U3_HNC10CT111'].values[i], median_filter_modified['U3_HNC10CT111'].values[i], ewma_modified['U3_HNC10CT111'].values[i]]
    #     df.loc[idx] = values
    #     idx += 1
    #
    # df.set_index('tid', inplace=True)

    # df.to_csv(PROJECT_ROOT + '/continuous.csv')
    # df.to_csv(PROJECT_ROOT + '/trend.csv')
    # df.to_csv(PROJECT_ROOT + '/reverse_trend.csv')
    # df.to_csv(PROJECT_ROOT + '/noise.csv')

    # df = pd.read_csv(PROJECT_ROOT + '/continuous.csv', index_col='tid')
    # df = pd.read_csv(PROJECT_ROOT + '/trend.csv', index_col='tid')
    # df = pd.read_csv(PROJECT_ROOT + '/reverse_trend.csv', index_col='tid')
    # df = pd.read_csv(PROJECT_ROOT + '/noise.csv', index_col='tid')

    # 画图设置
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'

    # 引言图
    # fig, ax = plt.subplots(figsize=(5, 3))

    # 画图样式
    # ax.plot(df.index, df['LP'].values, marker='H', linestyle='-', color='seagreen', label='LP', markersize=4, linewidth=1.5)
    # ax.plot(df.index, df['speed(L)'].values, marker='D', linestyle='-', color='red', label='Speed(L)', markersize=4, linewidth=1.5)
    # ax.plot(df.index, df['speed(G)'].values, marker='d', linestyle='-', color='royalblue', label='Speed(G)', markersize=4, linewidth=1.5)
    # ax.plot(df.index, df['IMR'].values, marker='x', linestyle='-', color='hotpink', label='IMR', markersize=4, linewidth=1.5)
    # ax.plot(df.index, df['Median'].values, marker='s', linestyle='-', color='gold', label='Median', markersize=3, linewidth=1.5)
    # ax.plot(df.index, df['origin'].values, marker='o', linestyle='-', color='black', label='origin', markersize=3, linewidth=1.5)

    # 另一种画图样式
    # ax.plot(df.index, df['LP'].values, marker='o', linestyle='-', color='orangered', label='MTSClean', markersize=3,
    #         linewidth=1.5)
    # ax.plot(df.index, df['speed(L)'].values, marker='o', linestyle='-', color='teal', label='Speed(L)', markersize=3,
    #         linewidth=1.5)
    # ax.plot(df.index, df['speed(G)'].values, marker='o', linestyle='-', color='deepskyblue', label='Speed(G)',
    #         markersize=3, linewidth=1.5)
    # ax.plot(df.index, df['IMR'].values, marker='o', linestyle='-', color='dodgerblue', label='IMR', markersize=3,
    #         linewidth=1.5)
    # ax.plot(df.index, df['Median'].values, marker='o', linestyle='-', color='royalblue', label='Median', markersize=3,
    #         linewidth=1.5)
    # ax.plot(df.index, df['origin'].values, marker='o', linestyle='-', color='black', label='origin', markersize=3,
    #         linewidth=1.5)

    # 可能需要的约束范围
    # ax.fill_between(df.index, idf.clean['U3_HNC10CT111'].values[25: 50], idf.clean['U3_HNV10CT111'].values[25: 50] * 0.163 + 47.642 + 0.085 + 0.25, label='bound', facecolor='grey', alpha=0.3)

    # ax.set_xlabel('timestamp')
    # ax.set_ylabel('U3_HNC10CT111')
    #
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # plt.savefig(PROJECT_ROOT + '/continuous.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    # plt.savefig(PROJECT_ROOT + '/trend.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    # plt.savefig(PROJECT_ROOT + '/reverse_trend.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    # plt.savefig(PROJECT_ROOT + '/noise.png', bbox_inches='tight', pad_inches=0.02, dpi=300)

    # plt.savefig(PROJECT_ROOT + '/continuous_color.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    # plt.savefig(PROJECT_ROOT + '/trend_color.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    # plt.savefig(PROJECT_ROOT + '/reverse_trend_color.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    # plt.savefig(PROJECT_ROOT + '/noise_color.png', bbox_inches='tight', pad_inches=0.02, dpi=300)

    # 绘制超图
    # t = array2window(idf.origin.values, win_size=w)[90]
    #
    # hypergraph_df = pd.DataFrame(columns=['edge', 'vertex', 'weight'])
    # edge_num = 1
    # for stcd in idf.stcds:
    #     degree = stcd.violation_degree(t)
    #     if degree > 0:
    #         for x_name in stcd.x_names:
    #             hypergraph_df.loc[len(hypergraph_df)] = ['e{}'.format(edge_num), x_name[1], degree]
    #         hypergraph_df.loc[len(hypergraph_df)] = ['e{}'.format(edge_num), stcd.y_name[1], degree]
    #         edge_num += 1
    #
    # H = hypernetx.Hypergraph(hypergraph_df, edge_col='edge', node_col='vertex', cell_weight_col='weight')
    # hypernetx.drawing.draw(H)
    # plt.show()
