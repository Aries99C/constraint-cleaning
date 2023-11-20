import random
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from utils import project_root
from dataset import DATASETS
from constraints.speed.mining import mining_speed_constraints
from constraints.acc.mining import mining_acc_constraints
from constraints.stcd.mining import mining_stcd

from cleaning.benchmark import delta, raa, speed_local, speed_global, acc_local, acc_global, IMR, ewma, median_filter, func_lp

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

    def __init__(self, dataset='idf', index_col='timestamp', datetime_index=True, size=5000, verbose=0):
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

    def constraints_mining(self, pre_mined=False, mining_constraints=None, w=2, verbose=0):
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
            if 'crr' in self.mining_constraints:    # 支持crr
                pass
            if 'rfd' in self.mining_constraints:    # 支持rfd
                pass
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
                self.stcds = mining_stcd(self, win_size=w, verbose=verbose)
                with open(PROJECT_ROOT + '/constraints/rules/{}_stcd.txt'.format(self.dataset), 'wb') as f:
                    pickle.dump(self.stcds, f)  # pickle序列化时窗约束
            if 'crr' in self.mining_constraints:    # 支持crr
                pass
            if 'rfd' in self.mining_constraints:    # 支持rfd
                pass

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
            insert_attrs = random.sample(self.cols, int(ratio * self.dim))  # 随机注入部分属性
            insert_len = random.randint(10, 50)                             # 随机的错误长度
            insert_pos = random.randint(20, self.len - insert_len - 1)      # 随机的注入位置
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

        # 初始化记录修复的单元格，False代表单元格没有修改
        self.isModified = self.origin.copy(deep=True)
        for col in self.isModified.columns:
            self.isModified[col] = False    # 全部初始化为False

        error_sum = self.delta_clean_origin()                           # 总误差
        error_aver = self.delta_clean_origin() / (self.len * self.dim)  # 平均误差

        print('{:=^80}'.format(' 向数据集{}注入错误 '.format(self.dataset.upper())))
        print('共计注入错误单元格数: {}'.format(insert_size))
        print('共计提供标签数：{}'.format(label_cnt))
        if verbose > 0:     # 日志显示
            # print('正确值和观测值的总误差: {:.3f}'.format(error_sum))
            print('注入错误的平均误差: {:.3f}'.format(error_aver))
            if verbose == 2:
                self.modified.plot(subplots=True, figsize=(20, 40))
                plt.show()

        return insert_size, error_sum, error_aver

    def delta_clean_origin(self):
        """
        计算观测值和正确值的距离
        """
        df = self.clean.sub(self.origin)    # 两个mts相减
        x = np.absolute(df.values)          # 获取ndarray的绝对值
        return np.sum(x)


if __name__ == '__main__':
    # 实验参数
    w = 2

    idf = MTS('idf', 'timestamp', True, size=5000, verbose=1)                   # 读取数据集
    # idf.constraints_mining(w=w, verbose=1)                                      # 挖掘约束
    idf.constraints_mining(pre_mined=True, verbose=1)                           # 预配置约束集合
    idf.insert_error(snr=15, verbose=1)                                         # 注入噪声

    # 速度约束Local修复
    # speed_local_modified, speed_local_is_modified, speed_local_time = speed_local(idf, w=w)
    # print('{:=^80}'.format(' 局部速度约束修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(speed_local_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(speed_local_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, speed_local_modified)))

    # 速度约束Global修复
    # speed_global_modified, speed_global_is_modified, speed_global_time = speed_global(idf, w=w, x=10)
    # print('{:=^80}'.format(' 全局速度约束修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(speed_global_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(speed_global_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, speed_global_modified)))

    # 速度约束+加速度约束Local修复
    # acc_local_modified, acc_local_is_modified, acc_local_time = acc_local(idf)
    # print('{:=^80}'.format(' 局部速度约束+加速度约束修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(acc_local_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(acc_local_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, acc_local_modified)))

    # 速度约束+加速度约束Global修复
    # acc_global_modified, acc_global_is_modified, acc_global_time = acc_global(idf)
    # print('{:=^80}'.format(' 全局速度约束+加速度约束修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(acc_global_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(acc_global_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, acc_global_modified)))

    # IMR修复
    # imr_modified, imr_is_modified, imr_time = IMR(max_iter=1000).clean(idf)
    # print('{:=^80}'.format(' IMR算法修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(imr_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(imr_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, imr_modified)))

    # EWMA修复
    # ewma_modified, ewma_is_modified, ewma_time = ewma(idf, beta=0.9)
    # print('{:=^80}'.format(' EWMA算法修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(ewma_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(ewma_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, ewma_modified)))

    # 中值滤波器修复
    # median_filter_modified, median_filter_is_modified, median_filter_time = median_filter(idf, w=10)
    # print('{:=^80}'.format(' 中值滤波修复数据集{} '.format(idf.dataset.upper())))
    # print('修复用时: {:.4g}ms'.format(median_filter_time))
    # print('修复值与正确值平均误差: {:.4g}'.format(delta(median_filter_modified, idf.clean)))
    # print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, median_filter_modified)))

    # func-LP修复
    func_lp_modified, func_lp_is_modified, func_lp_time = func_lp(idf, w=w)
    print('{:=^80}'.format(' func-LP修复数据集{} '.format(idf.dataset.upper())))
    print('修复用时: {:.4g}ms'.format(func_lp_time))
    print('修复值与正确值平均误差: {:.4g}'.format(delta(func_lp_modified, idf.clean)))
    print('修复相对精度: {:.4g}'.format(raa(idf.origin, idf.clean, func_lp_modified)))

    # idf.clean.plot(subplots=True, figsize=(10, 10))
    # idf.origin.plot(subplots=True, figsize=(10, 10))
    # func_lp_modified.plot(subplots=True, figsize=(10, 10))
    # plt.show()
