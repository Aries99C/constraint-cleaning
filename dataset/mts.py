import numpy as np
import pandas as pd

from utils import project_root
from dataset import DATASETS
from constraints.speed.mining import mining_speed_constraints
from constraints.acc.mining import mining_acc_constraints
from constraints.stcd.mining import mining_stcd

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

    def constraints_mining(self, pre_mined=False, mining_constraints=None, verbose=0):
        if mining_constraints is None:
            mining_constraints = ['speed', 'acc', 'stcd']   # 默认挖掘的约束
        if pre_mined:   # 使用预先挖掘好的规则，直接读取即可
            pass
        else:           # 否则需要挖掘规则
            if 'speed' in mining_constraints:   # 支持速度约束
                self.speed_constraints = mining_speed_constraints(self, alpha=3, verbose=verbose)
            if 'acc' in mining_constraints:     # 支持加速度约束
                self.acc_constraints = mining_acc_constraints(self, alpha=3, verbose=verbose)
            if 'stcd' in mining_constraints:    # 支持时窗约束
                self.stcds = mining_stcd(self, verbose=verbose)
            if 'crr' in mining_constraints:     # 支持crr
                pass
            if 'rfd' in mining_constraints:     # 支持rfd
                pass

    def clean2array(self):
        d = self.clean.copy(deep=True)      # 拷贝正确值
        return d.values                     # 将Dataframe转换为ndarray类型


if __name__ == '__main__':
    idf = MTS('idf', 'timestamp', True, size=5000, verbose=1)
    idf.constraints_mining(verbose=1)
