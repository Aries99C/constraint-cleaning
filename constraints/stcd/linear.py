import random
import numpy as np

from sklearn import linear_model


class Linear:
    """
    线性时窗约束的挖掘模型，提供线性时窗约束的规则挖掘与可视化功能
    """

    def __init__(self, mts, win_size=2, confidence=0.95, n_components=1, max_len=3,
                 sample_train=1.0, pc=1.0, ridge_all=False, implication_check=False):
        # 配置数据集
        self.mts = mts

        # 配置模型参数
        self.win_size = win_size
        self.confidence = confidence
        self.n_components = n_components
        self.max_len = max_len
        self.sample_train = sample_train
        self.pc = pc
        self.ridge_all = ridge_all
        self.implication_check = implication_check
        self.variables = []                         # 规则中的变量
        self.rules = []                             # 规则集合

        self.success_imp = 0
        self.all_imp = 0
        self.imp_time = 0

        # 配置规则中的变量名
        self.generate_variable_name()

    def generate_variable_name(self):
        self.variables = []
        for t in range(self.win_size):
            for i, col in enumerate(self.mts.cols):
                self.variables.append((t, col, i))   # 每个变量的形式为(时间戳，列名称，列索引)

    def mini_mine_sh(self, x, y, x_vars, y_var, verbose=0):
        # 生成随机掩码集合来学习规则
        x_possible = set()      # 可能存在关联的变量X集合
        while len(x_possible) < 4096:
            mask = self.random_mask(len(x_vars))    # 获得随机掩码
            if 1 not in mask:   # 随机选择没有选到任何变量X
                continue
            else:
                x_possible.add(mask.__str__())
        # 使用sklearn学习线性模型
        while len(x_possible) > self.n_components:  # 一直筛选直到得到参数限定的掩码个数
            n_possible = len(x_possible)
            mask_loss = {}                  # 记录随机掩码对应的拟合误差
            budget = int(x.shape[0] / (n_possible * np.log2(n_possible)) + 10)
            x_train = x[:budget, :]         # 训练数据x
            y_train = y[:budget]            # 训练数据y
            for mask_str in x_possible:     # 对每个掩码都学习一次线性函数
                mask = np.array(mask_str)   # 将掩码字符串转化为ndarray
                selected_x_train = None
                # 由掩码生成训练数据
                for idx in range(len(mask)):
                    if mask[idx] == 1:      # 选中掩码对应的列加入训练集
                        if selected_x_train is None:    # 第一次加入训练数据
                            selected_x_train = x_train[:, idx]
                        else:                           # 合并后加入的训练数据
                            selected_x_train = np.c_[selected_x_train, x_train[:, idx]]
                if len(selected_x_train) == 1:          # 如果只选中一列X，需要reshape成2d数据
                    selected_x_train = selected_x_train.reshape(-1, 1)
                # 训练线性模型
                model = linear_model.Ridge(self.pc)
                model.fit(selected_x_train, y_train)
                # 记录线性模型的拟合误差
                loss = np.sum((y_train - model.predict(selected_x_train)) ** 2)
                mask_loss[mask.__str__()] = loss
            # 根据误差重新筛选随机掩码
            sorted_mask_str = sorted(mask_loss.items(), key=lambda x: x[1], reverse=False)  # 根据误差升序排列掩码
            x_possible = set()
            while len(x_possible) < n_possible / 2:     # 删除一半的掩码
                x_possible.add(sorted_mask_str[len(x_possible)])
        # 使用筛选后的掩码生成规则
        # TODO

    def mine(self, verbose=0):
        if verbose > 0:         # 日志显示
            print('{:=^80}'.format(' 在数据集{}上挖掘线性时窗约束 '.format(self.mts.dataset.upper())))
        d = array2window(self.mts.clean2array(), self.win_size)     # 对多元时序数据切片
        if verbose > 0:         # 日志显示
            print('数据切片shape: {}'.format(d.shape))

        for i in range(d.shape[1]):     # 扫描每个切片的所有属性在时窗内多个时间戳上的值
            if verbose > 0:     # 日志显示
                print('{:=^40}'.format(' 挖掘Y=t{}[{}]上的f(X) '.format(self.variables[i][0], self.variables[i][1])))
            y_var = self.variables[i]   # 待挖掘函数Y=f(X)的变量Y
            x_vars = [var for var in self.variables if not (var[1] == y_var[1] and var[0] == y_var[0])]     # 变量X集合
            y = d[:, i]     # 变量Y的数据
            x = np.c_[d[:, 0: i], d[:, i+1:]] if i > 0 else d[:, i+1:]  # 变量X集合的数据
            # 挖掘时窗函数规则
            self.mini_mine_sh(x, y, x_vars, y_var, verbose=verbose)

    def random_mask(self, vec_len):
        """
        随机生成长度为vec_len的01掩码串
        :param vec_len: 生成的掩码长度
        :return: 长度为vec_len的随机01掩码串
        """
        mask = np.zeros(vec_len, dtype=int)
        cnt = 0
        up_bound = int(self.max_len * random.random())
        if up_bound == 0:
            up_bound = 1
        while cnt < up_bound:
            random_idx = np.random.randint(0, vec_len, dtype=int)
            if mask[random_idx] == 0:
                mask[random_idx] = 1
                cnt = cnt + 1
        return mask


def array2window(x, win_size=2):
    """
    将数据根据时窗长度切片。原本数据的shape为(n,m)，其中n为长度，m为维数，
    切片后得到的切片组shape为(n-win_size+1, win_size*m))。
    :param x: 原始数据，类型为ndarray，shape=(n,m)
    :param win_size: 时窗长度
    :return: 切片后的数据，类型为ndarray，shape=(n-win_size+1, win_size*m)
    """
    n, m = x.shape              # 原始数据shape
    slices = []                 # 返回的切片组
    for i in range(n - win_size + 1):                       # 共计n-win_size+1个窗口
        slices.append(x[i: i + win_size, :].reshape(-1))    # 切片
    slices = np.array(slices)   # 将list转换为ndarray
    return slices

