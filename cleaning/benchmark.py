import time
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import linprog
from constraints.stcd.linear import array2window


def delta(a, b, aver=True):
    df = a.sub(b)    # 两个mts相减
    x = np.absolute(df.values)  # 获取ndarray的绝对值
    if aver:
        x = x / (df.shape[0] * df.shape[1])
    return np.sum(x)


def raa(origin, clean, modified):
    return 1 - delta(modified, clean) / (delta(origin, modified) + delta(origin, clean))


def df2array(df):
    d = df.copy(deep=True)  # 拷贝值
    return d.values  # 将Dataframe转换为ndarray类型


def speed_local(mts, w=10):
    modified = mts.modified.copy(deep=True)         # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)    # 拷贝修复单元格信息
    time_cost = 0.                                  # 时间成本

    for col in modified.columns:    # 逐列修复
        # 获取当前列的速度约束上下界
        speed_lb = mts.speed_constraints[col][0]
        speed_ub = mts.speed_constraints[col][1]
        # 获取当前列的数据
        data = np.array(modified[col].values)
        # 构建速度约束问题
        start = time.perf_counter()
        for i in range(1, mts.len - 1):
            x_i_min = speed_lb + data[i - 1]
            x_i_max = speed_ub + data[i - 1]
            candidate_i = [data[i]]
            for k in range(i + 1, mts.len):
                if k > i + w:
                    break
                candidate_i.append(speed_lb + data[k])
                candidate_i.append(speed_ub + data[k])
            candidate_i = np.array(candidate_i)
            x_i_mid = np.median(candidate_i, axis=0)
            if x_i_min <= x_i_mid <= x_i_max:
                continue
            elif x_i_mid < x_i_min:
                modified[col].values[i] = x_i_min
                is_modified[col].values[i] = True
            else:
                modified[col].values[i] = x_i_max
                is_modified[col].values[i] = True
        end = time.perf_counter()
        time_cost += end - start

    return modified, is_modified, time_cost


def speed_global(mts, w=10, x=20, overlapping_ratio=0.2):
    modified = mts.modified.copy(deep=True)         # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)    # 拷贝修复单元格信息
    time_cost = 0.                                  # 时间成本

    for col in modified.columns:    # 逐列修复
        # 获取当前列的速度约束上下界
        speed_lb = mts.speed_constraints[col][0]
        speed_ub = mts.speed_constraints[col][1]

        s = 0
        while s + x <= mts.len:
            # 获取窗口x内的数据
            data = np.array(modified[col].values[s:min(s+x, mts.len)])
            # 构建线性规划问题
            start = time.perf_counter()
            c = np.ones(2 * data.shape[0])
            A = []
            b = []
            bounds = [(0, None) for j in range(2 * data.shape[0])]
            # 填补参数
            for i in range(data.shape[0]):
                for j in range(i + 1, data.shape[0]):
                    if j > i + w:
                        break
                    bij_min = -speed_lb * (j - i) + (data[j] - data[i])
                    bij_max = speed_ub * (j - i) - (data[j] - data[i])
                    b.append(bij_max)
                    b.append(bij_min)
                    aij_max = np.zeros(2 * data.shape[0])
                    aij_min = np.zeros(2 * data.shape[0])
                    aij_max[j], aij_max[i] = 1, -1
                    aij_max[j + data.shape[0]], aij_max[i + data.shape[0]] = -1, 1
                    A.append(aij_max)
                    aij_min[j], aij_min[i] = -1, 1
                    aij_min[j + data.shape[0]], aij_min[i + data.shape[0]] = 1, -1
                    A.append(aij_min)
            A = np.array(A)
            b = np.array(b)
            # 求解
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
            modified[col].values[s:min(s+x, mts.len)] = (res.x[:data.shape[0]] - res.x[data.shape[0]:]) + data
            s += int((1 - overlapping_ratio) * x)

            end = time.perf_counter()
            time_cost += end - start

    # 根据差值判断数据是否被修复
    for col in modified.columns:
        modified_values = modified[col].values
        origin_values = mts.origin[col].values
        for i in range(len(modified)):
            if abs(modified_values[i] - origin_values[i]) > 10e-3:
                is_modified[col].values[i] = True

    return modified, is_modified, time_cost


def acc_local(mts, w=10):
    modified = mts.modified.copy(deep=True)         # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)    # 拷贝修复单元格信息
    time_cost = 0.                                  # 时间成本

    for col in modified.columns:    # 逐列修复
        # 获取当前列的速度约束上下界
        speed_lb = mts.speed_constraints[col][0]
        speed_ub = mts.speed_constraints[col][1]
        # 获取当前列的加速度约束上下界
        acc_lb = mts.acc_constraints[col][0]
        acc_ub = mts.acc_constraints[col][1]
        # 获取当前列的数据
        data = np.array(modified[col].values)
        # 构建速度约束+加速度约束问题
        start = time.perf_counter()
        for k in range(2, mts.len - 1):
            candidate_k = [data[k]]
            candidate_k_min = []
            candidate_k_max = []
            x_k_min = max(speed_lb + data[k - 1], acc_lb + data[k - 1] - data[k - 2] + data[k - 1])
            x_k_max = min(speed_ub + data[k - 1], acc_ub + data[k - 1] - data[k - 2] + data[k - 1])
            for i in range(k + 1, mts.len):
                if i > k + w:
                    break
                z_k_i_a_min = (data[k - 1] * (i - k) - (acc_ub * ((i - k) ** 2) - data[i])) / (i - k + 1)
                z_k_i_a_max = (data[k - 1] * (i - k) - (acc_lb * ((i - k) ** 2) - data[i])) / (i - k + 1)
                z_k_i_s_min = data[i] - speed_ub * (i - k)
                z_k_i_s_max = data[i] - speed_lb * (i - k)
                candidate_k_min.append(min(z_k_i_s_min, z_k_i_a_min))
                candidate_k_max.append(max(z_k_i_s_max, z_k_i_a_max))
            candidate_k = np.array(candidate_k + candidate_k_min + candidate_k_max)
            x_k_mid = np.median(candidate_k, axis=0)
            if x_k_min <= x_k_mid <= x_k_max:
                continue
            elif x_k_mid < x_k_min:
                modified[col].values[k] = x_k_min
                is_modified[col].values[k] = True
            else:
                modified[col].values[k] = x_k_max
                is_modified[col].values[k] = True
        end = time.perf_counter()
        time_cost += end - start

    return modified, is_modified, time_cost


def acc_global(mts, w=10, x=20, overlapping_ratio=0.2):
    modified = mts.modified.copy(deep=True)         # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)    # 拷贝修复单元格信息
    time_cost = 0.                                  # 时间成本

    for col in modified.columns:    # 逐列修复
        # 获取当前列的速度约束上下界
        speed_lb = mts.speed_constraints[col][0]
        speed_ub = mts.speed_constraints[col][1]
        # 获取当前列的加速度约束上下界
        acc_lb = mts.acc_constraints[col][0]
        acc_ub = mts.acc_constraints[col][1]

        s = 0
        while s + x <= mts.len:
            # 获取窗口x内的数据
            data = np.array(modified[col].values[s:min(s+x, mts.len)])
            # 构建线性规划问题
            start = time.perf_counter()
            c = np.ones(2 * data.shape[0])
            A = []
            b = []
            bounds = [(0, None) for j in range(2 * data.shape[0])]
            # 填补参数
            for i in range(data.shape[0]):
                for j in range(i + 1, data.shape[0]):
                    if j > i + w:
                        break
                    bij_min = -speed_lb * (j - i) + (data[j] - data[i])
                    bij_max = speed_ub * (j - i) - (data[j] - data[i])
                    b.append(bij_max)
                    b.append(bij_min)
                    aij_max = np.zeros(2 * data.shape[0])
                    aij_min = np.zeros(2 * data.shape[0])
                    aij_max[j], aij_max[i] = 1, -1
                    aij_max[j + data.shape[0]], aij_max[i + data.shape[0]] = -1, 1
                    A.append(aij_max)
                    aij_min[j], aij_min[i] = -1, 1
                    aij_min[j + data.shape[0]], aij_min[i + data.shape[0]] = 1, -1
                    A.append(aij_min)
                    if i >= 1:
                        bij_max = acc_ub * (j - i) - (data[j] - data[i]) / (j - i) + (data[i] - data[i - 1])
                        tmp1, tmp2 = 1 / (j - i), 1
                        aij_max = np.zeros(2 * data.shape[0])
                        aij_max[j], aij_max[j + data.shape[0]] = tmp1, -tmp1
                        aij_max[i], aij_max[i + data.shape[0]] = -tmp1 - tmp2, tmp1 + tmp2
                        aij_max[i - 1], aij_max[i - 1 + data.shape[0]] = tmp2, -tmp2
                        b.append(bij_max)
                        A.append(aij_max)
                        aij_min = np.zeros(2 * data.shape[0])
                        bij_min = -acc_lb * (j - i) + (data[j] - data[i]) / (j - i) - (data[i] - data[i - 1])
                        aij_min[j], aij_min[j + data.shape[0]] = -tmp1, tmp1
                        aij_min[i], aij_min[i + data.shape[0]] = tmp1 + tmp2, -tmp1 - tmp2
                        aij_min[i - 1], aij_min[i - 1 + data.shape[0]] = -tmp2, tmp2
                        b.append(bij_min)
                        A.append(aij_min)
            A = np.array(A)
            b = np.array(b)
            # 求解
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
            modified[col].values[s:min(s+x, mts.len)] = (res.x[:data.shape[0]] - res.x[data.shape[0]:]) + data
            s += int((1 - overlapping_ratio) * x)

            end = time.perf_counter()
            time_cost += end - start

    # 根据差值判断数据是否被修复
    for col in modified.columns:
        modified_values = modified[col].values
        origin_values = mts.origin[col].values
        for i in range(len(modified)):
            if abs(modified_values[i] - origin_values[i]) > 10e-3:
                is_modified[col].values[i] = True

    return modified, is_modified, time_cost


class IMR:
    def __init__(self, p=3, threshold=0.1, max_iter=1000):
        # IMR模型参数
        self.p = p
        self.threshold = threshold
        self.max_iter = max_iter
        self.MIN_VAL = np.inf
        self.MAX_VAL = -np.inf
        # IMR使用的标签信息
        self.label_list = None
        self.labels = None

    def learnParamsOLS(self, x_matrix, y_matrix):
        phi = np.zeros((self.p, 1))

        x_matrix_t = x_matrix.T

        middle_matrix = x_matrix_t.dot(x_matrix)
        phi = np.linalg.pinv(middle_matrix).dot(x_matrix_t).dot(y_matrix)

        return phi

    def combine(self, phi, x_matrix):
        yhat_matrix = x_matrix.dot(phi)

        return yhat_matrix

    def repairAMin(self, yhat_matrix, y_matrix):
        row_num = yhat_matrix.shape[0]
        residual_matrix = yhat_matrix - y_matrix

        a_min = self.MIN_VAL
        target_index = -1
        yhat = None
        yhat_abs = None

        for i in range(row_num):
            if self.label_list[i + self.p]:
                continue
            if abs(residual_matrix[i, 0]) < self.threshold:
                continue

            yhat = yhat_matrix[i, 0]
            yhat_abs = abs(yhat)

            if yhat_abs < a_min:
                a_min = yhat_abs
                target_index = i

        return target_index

    def clean(self, mts):
        modified = mts.modified.copy(deep=True)         # 拷贝数据
        is_modified = mts.isModified.copy(deep=True)    # 拷贝修复单元格信息
        time_cost = 0.                                  # 时间成本

        for col in modified.columns:
            # 获取标注信息
            self.label_list = mts.isLabel[col].values
            self.labels = mts.clean[col].values
            # 获取待修复序列
            data = modified[col].values

            start = time.perf_counter()

            size = len(data)
            row_num = size - self.p

            # form z
            zs = []
            for i in range(size):
                zs.append(self.labels[i] - data[i])

            # build x,y for params estimation
            x = np.zeros((row_num, self.p))
            y = np.zeros((row_num, 1))
            for i in range(row_num):
                y[i, 0] = zs[self.p + i]
                for j in range(self.p):
                    x[i, j] = zs[self.p + i - j - 1]

            # iteration
            index = -1
            x_matrix = np.matrix(x)
            y_matrix = np.matrix(y)
            iteration_num = 0
            val = 0

            phi = None
            while True:
                iteration_num += 1

                phi = self.learnParamsOLS(x_matrix, y_matrix)

                yhat_matrix = self.combine(phi, x_matrix)

                index = self.repairAMin(yhat_matrix, y_matrix)

                if index == -1:
                    break

                val = yhat_matrix[index, 0]
                # update y
                y_matrix[index, 0] = val
                # update x
                for j in range(self.p):
                    i = index + 1 + j
                    if i >= row_num:
                        break
                    if i < 0:
                        continue

                    x_matrix[i, j] = val

                # 迭代控制
                if iteration_num > self.max_iter:
                    break

            # 修复
            for i in range(size):
                if self.label_list[i]:
                    data[i] = self.labels[i]
                else:
                    data[i] = data[i] + y_matrix[i - self.p, 0]

            end = time.perf_counter()
            time_cost += end - start

        # 根据差值判断数据是否被修复
        for col in modified.columns:
            modified_values = modified[col].values
            origin_values = mts.origin[col].values
            for i in range(len(modified)):
                if abs(modified_values[i] - origin_values[i]) > 10e-3:
                    is_modified[col].values[i] = True

        return modified, is_modified, time_cost


def ewma(mts, beta=0.9):
    modified = mts.modified.copy(deep=True)         # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)    # 拷贝修复单元格信息
    time_cost = 0.                                  # 时间成本

    for col in modified.columns:
        data = modified[col].values
        start = time.perf_counter()
        for i in range(1, len(data)):
            data[i] = beta * data[i - 1] + (1 - beta) * data[i]
        end = time.perf_counter()
        time_cost += end - start

    # 根据差值判断数据是否被修复
    for col in modified.columns:
        modified_values = modified[col].values
        origin_values = mts.origin[col].values
        for i in range(len(modified)):
            if abs(modified_values[i] - origin_values[i]) > 10e-3:
                is_modified[col].values[i] = True

    return modified, is_modified, time_cost


def median_filter(mts, w=10):
    modified = mts.modified.copy(deep=True)         # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)    # 拷贝修复单元格信息
    time_cost = 0.                                  # 时间成本

    for col in modified.columns:
        data = modified[col].values
        start = time.perf_counter()
        for i in range(int(w / 2), len(data) - int(w / 2)):
            data[i] = np.median(data[i - int(w / 2): i - int(w / 2) + w])
        end = time.perf_counter()
        time_cost += end - start

    # 根据差值判断数据是否被修复
    for col in modified.columns:
        modified_values = modified[col].values
        origin_values = mts.origin[col].values
        for i in range(len(modified)):
            if abs(modified_values[i] - origin_values[i]) > 10e-3:
                is_modified[col].values[i] = True

    return modified, is_modified, time_cost


def func_lp(mts, w=2):
    modified = mts.modified.copy(deep=True)         # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)    # 拷贝修复单元格信息
    time_cost = 0.                                  # 时间成本

    for i in range(mts.len - w):    # 对每个切片直接构建线性规划问题
        # 获取切片数据
        data = array2window(df2array(modified), w)[i]

        # 构建线性规划问题
        start = time.perf_counter()
        c = np.ones(2 * data.shape[0])  # u_i和v_i的系数均为1，目标函数：min Σ(u_i + v_i)
        A = []  # u_i和v_i的系数由stcd中每个t_i[A]的系数得到
        b = []  # b由stcd的lb和ub得到
        bounds = [(0, None) for j in range(2 * data.shape[0])]  # 所有u_i和v_i都大于等于0

        # 添加速度约束
        for j, col in enumerate(mts.cols):
            # 获取当前列的速度约束上下界
            s_lb = mts.speed_constraints[col][0]
            s_ub = mts.speed_constraints[col][1]
            for l in range(w):
                for r in range(l + 1, w):
                    # 速度约束下界
                    b_lr_min = -s_lb * (r - l) + (data[l * mts.dim + j] - data[r * mts.dim + j])
                    a_lr_min = np.zeros(2 * data.shape[0])
                    a_lr_min[r * mts.dim + j], a_lr_min[l * mts.dim + j] = 1, -1
                    a_lr_min[r * mts.dim + j + data.shape[0]], a_lr_min[l * mts.dim + j + data.shape[0]] = -1, 1
                    A.append(a_lr_min)
                    b.append(b_lr_min)
                    # 速度约束上界
                    b_lr_max = s_ub * (r - l) + (data[l * mts.dim + j] - data[r * mts.dim + j])
                    a_lr_max = np.zeros(2 * data.shape[0])
                    a_lr_max[r * mts.dim + j], a_lr_max[l * mts.dim + j] = 1, -1
                    a_lr_max[r * mts.dim + j + data.shape[0]], a_lr_max[l * mts.dim + j + data.shape[0]] = -1, 1
                    A.append(a_lr_max)
                    b.append(b_lr_max)

        if w > 1:
            res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds).x  # 默认是速度约束解
        else:
            res = None

        for rule in mts.stcds:  # 将stcd转化为线性规划问题的约束条件
            lb = rule.lb    # 约束下界
            ub = rule.ub    # 约束上界
            y_pos = rule.y_name[0] * mts.dim + rule.y_name[2]   # 约束f(X)->Y的Y在切片中的位置
            # 根据y_pos将切片分为x和y
            x = np.append(data[:y_pos], data[y_pos+1:]) if y_pos > 0 else data[y_pos+1:]
            y = data[y_pos]
            # 计算约束对应的系数A和b
            # 将x'和y'都变成u和v：x'=x+(u-v), y'=y+(u-v);

            # 上界对应约束 Σ a_i * x'_i + b - y' <= ub
            a_ub = np.zeros(2 * data.shape[0])          # 前一半是u_i的系数，后一半是v_i的系数
            b_ub = ub + y
            for j in range(len(rule.x_names)):          # 变量x在x_names中的顺序
                x_pos = rule.x_names[j][0] * mts.dim + rule.x_names[j][2]   # 变量x在切片中的位置
                x_coef = rule.func['coef'][j]           # 变量x在stcd中的系数
                a_ub[x_pos], a_ub[x_pos + data.shape[0]] = x_coef, -x_coef  # u_x和v_x的系数为a_x和-a_x
                b_ub -= x_coef * x[j]                   # 每个原变量x'都提供一个x，需要在上界中减去
            a_ub[y_pos], a_ub[y_pos + data.shape[0]] = -1, 1        # u_y和v_y的系数为-1和1
            # 添加约束
            A.append(a_ub)
            b.append(b_ub)

            # 下界对应约束 lb <= Σ a_i * x'_i + b - y'
            # 转化成小于等于的形式：Σ -a_i * x'_i
            a_lb = np.zeros(2 * data.shape[0])      # 前一半是u_i的系数，后一半是v_i的系数
            b_lb = - lb - y
            for j in range(len(rule.x_names)):          # 变量x在x_names中的顺序
                x_pos = rule.x_names[j][0] * mts.dim + rule.x_names[j][2]   # 变量x在切片中的位置
                x_coef = rule.func['coef'][j]           # 变量x在stcd中的系数
                a_lb[x_pos], a_lb[x_pos + data.shape[0]] = -x_coef, x_coef  # u_x和v_x的系数为a_x和-a_x
                b_lb += x_coef * x[j]                   # 每个原变量x'都提供一个x，需要在上界中减去
            a_ub[y_pos], a_ub[y_pos + data.shape[0]] = 1, -1        # u_y和v_y的系数为-1和1
            # 添加约束
            A.append(a_lb)
            b.append(b_lb)
            # 记录最后一次成功求解
            tmp = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds).x
            if tmp is not None:
                res = tmp
            else:
                break
        # 记录运行时间
        end = time.perf_counter()
        time_cost += end - start

        # 求解成功则更新数据
        if res is not None:
            modified.values[i:i+w, :] = (res[w*mts.dim:] - res[:w*mts.dim] + data).reshape(w, mts.dim)

    # 根据差值判断数据是否被修复
    for col in modified.columns:
        modified_values = modified[col].values
        origin_values = mts.origin[col].values
        for i in range(len(modified)):
            if abs(modified_values[i] - origin_values[i]) > 10e-3:
                is_modified[col].values[i] = True

    return modified, is_modified, time_cost
