import random
import time
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import linprog
from constraints.stcd.linear import array2window
from sklearn.metrics import precision_score, recall_score, f1_score


def delta(a, b, aver=True):
    df = a.sub(b)  # 两个mts相减
    x = np.absolute(df.values)  # 获取ndarray的绝对值
    if aver:
        x = x / (df.shape[0] * df.shape[1])
    return np.sum(x)


def raa(origin, clean, modified):
    return 1 - delta(modified, clean) / (delta(origin, modified) + delta(origin, clean))


def f1(is_modified, is_dirty, average='binary'):
    a = is_modified.values.reshape(-1)
    b = is_dirty.values.reshape(-1)

    p, r, f = precision_score(b, a, average=average), recall_score(b, a, average=average), f1_score(b, a, average=average)

    return p, r, f


def check_repair_violation(modified, rules, w):
    slices = array2window(df2array(modified), w)

    violation_rate = 0.

    for t in slices:
        for rule in rules:
            score = rule.violation_degree(t)
            violation_rate += score / (rule.ub + 1e-4)

    violation_rate /= (len(modified) * len(modified.columns))

    return violation_rate


def violation_rate(modified, rules, w):
    slices = array2window(df2array(modified), w)

    violation_rate = 0.

    for t in slices:
        for rule in rules:
            if rule.violation_degree(t) > 1e-5:
                violation_rate += 1

    violation_rate /= ((len(modified) - w) * len(rules))

    return violation_rate

def df2array(df):
    d = df.copy(deep=True)  # 拷贝值
    return d.values  # 将Dataframe转换为ndarray类型


def update_is_modified(mts, modified, is_modified):
    for col in modified.columns:
        modified_values = modified[col].values
        origin_values = mts.origin[col].values
        for i in range(len(modified)):
            if abs(modified_values[i] - origin_values[i]) > 10e-3:
                is_modified[col].values[i] = True


def speed_local(mts, w=10):
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本

    for col in modified.columns:  # 逐列修复
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
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本

    for col in modified.columns:  # 逐列修复
        # 获取当前列的速度约束上下界
        speed_lb = mts.speed_constraints[col][0]
        speed_ub = mts.speed_constraints[col][1]

        s = 0
        while s + x <= mts.len:
            # 获取窗口x内的数据
            data = np.array(modified[col].values[s:min(s + x, mts.len)])
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
            modified[col].values[s:min(s + x, mts.len)] = (res.x[:data.shape[0]] - res.x[data.shape[0]:]) + data
            s += int((1 - overlapping_ratio) * x)

            end = time.perf_counter()
            time_cost += end - start

    # 根据差值判断数据是否被修复
    update_is_modified(mts, modified, is_modified)

    return modified, is_modified, time_cost


def acc_local(mts, w=10):
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本

    for col in modified.columns:  # 逐列修复
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
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本

    for col in modified.columns:  # 逐列修复
        # 获取当前列的速度约束上下界
        speed_lb = mts.speed_constraints[col][0]
        speed_ub = mts.speed_constraints[col][1]
        # 获取当前列的加速度约束上下界
        acc_lb = mts.acc_constraints[col][0]
        acc_ub = mts.acc_constraints[col][1]

        s = 0
        while s + x <= mts.len:
            # 获取窗口x内的数据
            data = np.array(modified[col].values[s:min(s + x, mts.len)])
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
            modified[col].values[s:min(s + x, mts.len)] = (res.x[:data.shape[0]] - res.x[data.shape[0]:]) + data
            s += int((1 - overlapping_ratio) * x)

            end = time.perf_counter()
            time_cost += end - start

    # 根据差值判断数据是否被修复
    update_is_modified(mts, modified, is_modified)

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
        modified = mts.modified.copy(deep=True)  # 拷贝数据
        is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
        time_cost = 0.  # 时间成本

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
        update_is_modified(mts, modified, is_modified)

        return modified, is_modified, time_cost


def ewma(mts, beta=0.9):
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本

    for col in modified.columns:
        data = modified[col].values
        start = time.perf_counter()
        for i in range(1, len(data)):
            data[i] = beta * data[i - 1] + (1 - beta) * data[i]
        end = time.perf_counter()
        time_cost += end - start

    # 根据差值判断数据是否被修复
    update_is_modified(mts, modified, is_modified)

    return modified, is_modified, time_cost


def median_filter(mts, w=10):
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本

    for col in modified.columns:
        data = modified[col].values
        start = time.perf_counter()
        for i in range(int(w / 2), len(data) - int(w / 2)):
            data[i] = np.median(data[i - int(w / 2): i - int(w / 2) + w])
        end = time.perf_counter()
        time_cost += end - start

    # 根据差值判断数据是否被修复
    update_is_modified(mts, modified, is_modified)

    return modified, is_modified, time_cost


def func_lp(mts, w=2, size=20, overlapping_ratio=0.2):
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本
    success_cnt = 0
    win_num = 0

    m = mts.dim  # 属性个数

    s = 0
    while s + size <= mts.len:
        win_num += 1    # 记录清洗了多少个窗口
        # 获取大窗口内的数据
        win = min(size, mts.len - s)
        data = array2window(df2array(modified), win)[s]
        # 构建线性规划问题
        start = time.perf_counter()
        # 初始化
        c = np.ones(2 * data.shape[0])
        A = []
        b = []
        bounds = [(0, None) for j in range(2 * data.shape[0])]

        # 利用速度约束填补参数
        for idx, col in enumerate(modified.columns):
            # 获取当前列的速度约束上下界
            s_lb = mts.speed_constraints[col][0]
            s_ub = mts.speed_constraints[col][1]
            # 时窗限制内约束
            for i in range(win):
                for j in range(i + 1, win):
                    if j > i + w:  # 只考虑时窗限制内的
                        break
                    # 速度下界
                    b_lb = -s_lb * (j - i) + (data[j * m + idx] - data[i * m + idx])
                    a_lb = np.zeros(2 * data.shape[0])
                    a_lb[j * m + idx], a_lb[j * m + idx + data.shape[0]] = -1, 1
                    a_lb[i * m + idx], a_lb[i * m + idx + data.shape[0]] = 1, -1
                    A.append(a_lb)
                    b.append(b_lb)
                    # 速度上界
                    b_ub = s_ub * (j - i) - (data[j * m + idx] - data[i * m + idx])
                    a_ub = np.zeros(2 * data.shape[0])
                    a_ub[j * m + idx], a_ub[j * m + idx + data.shape[0]] = 1, -1
                    a_ub[i * m + idx], a_ub[i * m + idx + data.shape[0]] = -1, 1
                    A.append(a_ub)
                    b.append(b_ub)
        # # 默认速度约束求解
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds).x
        # res = None

        rules = sorted(mts.stcds, key=lambda r: r.ub - r.lb, reverse=True)[:20]
        for rule in rules:
            # 尝试添加约束前先保存上一次添加成功的系数
            A_old = A.copy()
            b_old = b.copy()
            # 解析约束的参数
            lb = rule.lb  # 约束下界
            ub = rule.ub  # 约束上界
            intercept = rule.func['intercept']  # 约束的模型截距
            y_pos = rule.y_name[0] * m + rule.y_name[2]  # 约束f(X)->Y的Y在切片中的位置
            # 对每个切片生成一组约束
            for i in range(win - w):
                # 获取切片
                t = data[i * m: (i + w) * m]

                # 根据y_pos将切片分为x和y
                x = np.append(t[:y_pos], t[y_pos + 1:]) if y_pos > 0 else t[y_pos + 1:]
                y = t[y_pos]
                # 计算约束对应的系数A和b
                # 将x'和y'都变成u和v：x'=x+(u-v), y'=y+(u-v);

                # 上界对应约束 Σ a_i * x'_i + b - y' <= ub
                a_ub = np.zeros(2 * data.shape[0])  # 前一半是u_i的系数，后一半是v_i的系数
                b_ub = ub + y - intercept
                a_ub[i * m: (i + w) * m] = rule.alpha
                a_ub[i * m + data.shape[0]: (i + w) * m + data.shape[0]] = -rule.alpha
                for j in range(len(t)):
                    b_ub -= rule.alpha[j] * t[j]
                A.append(a_ub)
                b.append(b_ub)

                # 下界
                a_lb = np.zeros(2 * data.shape[0])  # 前一半是u_i的系数，后一半是v_i的系数
                b_lb = -lb - y + intercept
                a_lb[i * m: (i + w) * m] = -rule.alpha
                a_lb[i * m + data.shape[0]: (i + w) * m + data.shape[0]] = rule.alpha
                for j in range(len(t)):
                    b_lb += rule.alpha[j] * t[j]
                A.append(a_lb)
                b.append(b_lb)

            try_x = linprog(c, A_ub=A, b_ub=b, bounds=bounds).x
            if try_x is not None:
                success_cnt += 1
                res = try_x
            else:
                print('添加约束{}失败，该约束的上下界为({},{})，变量X的系数为{}'.format(rule, rule.lb, rule.ub, rule.func['coef']))
                A = A_old
                b = b_old

        end = time.perf_counter()
        time_cost += end - start

        if res is not None:
            modified.values[s:s + win] = ((res[:data.shape[0]] - res[data.shape[0]:]) + data).reshape(win, m)
        s += int((1 - overlapping_ratio) * size)

    # 根据差值判断数据是否被修复
    update_is_modified(mts, modified, is_modified)

    # print('平均成功添加{:.2f}%的约束'.format(success_cnt / 20 / win_num * 100))
    return modified, is_modified, time_cost


def func_mvc(mts, w=2, size=20, mvc='sorted', overlapping_ratio=0.3):
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本
    success_cnt = 0
    lp_size = 0.

    m = mts.dim  # 属性个数

    s = 0
    while s + size <= mts.len:
        # 获取大窗口内的数据
        win = min(size, mts.len - s)
        data = array2window(df2array(modified), win)[s]
        # 构建线性规划问题
        start = time.perf_counter()
        # 初始化
        c = np.ones(2 * data.shape[0])
        A = []
        b = []
        bounds = [(0, None) for j in range(2 * data.shape[0])]
        # 利用速度约束填补参数
        for idx, col in enumerate(modified.columns):
            # 获取当前列的速度约束上下界
            s_lb = mts.speed_constraints[col][0]
            s_ub = mts.speed_constraints[col][1]
            # 时窗限制内约束
            for i in range(win):
                for j in range(i + 1, win):
                    if j > i + w:  # 只考虑时窗限制内的
                        break
                    # 速度下界
                    b_lb = -s_lb * (j - i) + (data[j * m + idx] - data[i * m + idx])
                    a_lb = np.zeros(2 * data.shape[0])
                    a_lb[j * m + idx], a_lb[j * m + idx + data.shape[0]] = -1, 1
                    a_lb[i * m + idx], a_lb[i * m + idx + data.shape[0]] = 1, -1
                    A.append(a_lb)
                    b.append(b_lb)
                    # 速度上界
                    b_ub = s_ub * (j - i) - (data[j * m + idx] - data[i * m + idx])
                    a_ub = np.zeros(2 * data.shape[0])
                    a_ub[j * m + idx], a_ub[j * m + idx + data.shape[0]] = 1, -1
                    a_ub[i * m + idx], a_ub[i * m + idx + data.shape[0]] = -1, 1
                    A.append(a_ub)
                    b.append(b_ub)
        # 默认速度约束求解
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds).x

        for i in range(win - w):
            # 获取切片构建超图
            t = data[i * m: (i + w) * m]
            hypergraph = (t, mts.stcds)

            # 切片做约束违反检测
            violation = violation_detect(hypergraph)

            # 调用FindKeyCell找到切片中的关键单元格
            repair_edge, key_cell_pos = find_key_cell(violation, mvc=mvc)

            lp_size += len(repair_edge)

            # 围绕关键单元格生成约束
            for rule in repair_edge:
                # 解析约束的参数
                lb = rule.lb  # 约束下界
                ub = rule.ub  # 约束上界
                intercept = rule.func['intercept']  # 约束的模型截距
                y_pos = rule.y_name[0] * m + rule.y_name[2]  # 约束f(X)->Y的Y在切片中的位置
                # 根据y_pos将切片分为x和y
                x = np.append(t[:y_pos], t[y_pos + 1:]) if y_pos > 0 else t[y_pos + 1:]
                y = t[y_pos]
                # 计算约束对应的系数A和b
                # 将x'和y'都变成u和v：x'=x+(u-v), y'=y+(u-v);

                # 上界对应约束 Σ a_i * x'_i + b - y' <= ub
                a_ub = np.zeros(2 * data.shape[0])  # 前一半是u_i的系数，后一半是v_i的系数
                b_ub = ub + y - intercept
                a_ub[i * m: (i + w) * m] = rule.alpha
                a_ub[i * m + data.shape[0]: (i + w) * m + data.shape[0]] = -rule.alpha
                for j in range(len(t)):
                    b_ub -= rule.alpha[j] * t[j]
                A.append(a_ub)
                b.append(b_ub)

        try_x = linprog(c, A_ub=A, b_ub=b, bounds=bounds).x
        if try_x is not None:
            success_cnt += 1
            res = try_x

        end = time.perf_counter()
        time_cost += end - start

        if res is not None:
            modified.values[s:s + win] = ((res[:data.shape[0]] - res[data.shape[0]:]) + data).reshape(win, m)
        s += int((1 - overlapping_ratio) * size)

    # 根据差值判断数据是否被修复
    update_is_modified(mts, modified, is_modified)

    print('构建问题所用约束数量: {:.4g}'.format(lp_size / len(mts.stcds) / mts.len))

    return modified, is_modified, time_cost


def violation_detect(hypergraph):
    t, rules = hypergraph  # 解析超图

    violation = []  # 约束违反情况

    for rule in rules:
        # 调用规则的违反程度分数计算
        degree = rule.violation_degree(t)
        if degree >= 1e-3:  # 没有被违反的规则可以删除
            violation.append((rule, degree))

    return violation


def find_key_cell(violation, mvc='sorted'):
    key_cell_pos = set()   # 最终返回待清洗的关键变量
    repair_edge = []    # 最终返回清洗所用约束

    if mvc == 'sorted':
        pre_violation = sorted(violation, key=lambda x: x[1], reverse=True)

        repair_edge = []

        for rule, degree in pre_violation:
            # 跳过已经被删除的边
            skip = False
            for pos in key_cell_pos:
                if not rule.alpha[pos] == 0:
                    skip = True
                    break
            if skip:
                continue

            # 将新的边删除
            repair_edge.append(rule)
            for i, x in enumerate(rule.alpha):
                if not x == 0:
                    key_cell_pos.add(i)

    if mvc == 'shuffle':
        random.shuffle(violation)

        repair_edge = []

        for rule, degree in violation:
            # 跳过已经被删除的边
            skip = False
            for pos in key_cell_pos:
                if not rule.alpha[pos] == 0:
                    skip = True
                    break
            if skip:
                continue

            # 将新的边删除
            repair_edge.append(rule)
            for i, x in enumerate(rule.alpha):
                if not x == 0:
                    key_cell_pos.add(i)

    if mvc == 'max_degree':
        n = len(violation[0][0].alpha)     # 所有pos的总长度

        while len(violation) > 0:       # 一直删除直至所有的边都被覆盖
            # 寻找当前图中度数最大的顶点
            max_pos = -1
            max_degree = 0
            for i in range(n):          # 计算每个pos的度数
                if i in key_cell_pos:   # 跳过已经被加入Cover中的pos
                    continue
                cur_degree = 0
                for j, vio in enumerate(violation):     # 判断每条未删除的边是否与当前pos连接
                    if not vio[0].alpha[i] == 0:      # 当前边还未删除且与该pos连接
                        cur_degree += 1     # 当前pos的度数+1
                if cur_degree > max_degree:     # 不断更新度数最大的pos
                    max_degree = cur_degree
                    max_pos = i

            # 删除度数最大的pos与其连接的边
            key_cell_pos.add(max_pos)

            max_violation_degree = 0.
            max_rule = None

            t = violation[:]
            for vio in violation:
                if not vio[0].alpha[max_pos] == 0:
                    if vio[1] > max_violation_degree:
                        max_violation_degree = vio[1]
                        max_rule = vio[0]
                    t.remove(vio)
            violation = t

            if max_rule is not None:
                repair_edge.append(max_rule)

    if mvc == 'vertex_support':
        n = len(violation[0][0].alpha)  # 所有pos的总长度
        m = len(violation)

        rest_edge = set(range(m))   # 剩余的边索引集合
        rest_pos = set(range(n))

        while len(rest_edge) > 0:  # 一直删除直至所有的边都被覆盖
            # 先计算所有顶点的度数
            degrees = np.zeros(n)
            for i, vio in enumerate(violation):
                if i in rest_edge:  # 从剩余的边中计算度数
                    degrees = degrees + np.array([1 if not x == 0 else 0 for x in vio[0].alpha])

            # 再计算所有顶点的支持度
            supports = np.zeros(n)
            for k, vio in enumerate(violation):
                if k in rest_edge:  # 从剩余的边中判断邻居
                    alpha = vio[0].alpha
                    for i in range(n):
                        for j in range(n):
                            if (not i == j) and (not alpha[i] == 0) and (not alpha[j] == 0):    # 顶点i和j是邻居
                                supports[i] += degrees[j]
                                supports[j] += degrees[i]

            # 寻找最大的支持度对应的顶点u
            max_support_pos = np.argmax(supports)

            # 从图中删除顶点u以及与其相连的边
            # 先删除边，同时找到违反程度最高的作为修复用边
            max_violation_degree = 0.
            max_violation_edge = None

            to_delete = set()
            for i, vio in enumerate(violation):
                if i in rest_edge:
                    if not vio[0].alpha[max_support_pos] == 0:      # 当前边与顶点u连接
                        to_delete.add(i)    # 添加待删除的边的索引
                        if vio[1] > max_violation_degree:
                            max_violation_degree = vio[1]
                            max_violation_edge = vio[0]
            # 更新剩余边的索引集合
            rest_edge = rest_edge - to_delete
            rest_pos = rest_pos - {max_support_pos}
            # 再删除顶点u
            key_cell_pos.add(max_support_pos)

            if max_violation_edge is not None:
                repair_edge.append(max_violation_edge)

    if mvc == 'greedy':
        n = len(violation[0][0].alpha)  # 所有pos的总长度
        m = len(violation)

        rest_edge = set(range(m))  # 剩余的边索引集合
        rest_pos = set(range(n))

        while len(rest_edge) > 0:  # 一直删除直至所有的边都被覆盖
            # 先计算所有顶点的度数
            degrees = np.zeros(n)
            for i, vio in enumerate(violation):
                if i in rest_edge:  # 从剩余的边中计算度数
                    degrees = degrees + np.array([1 if not x == 0 else 0 for x in vio[0].alpha])
            # 寻找最小的度数对应的顶点u
            min_degree_pos = np.argmin([degrees[i] if i in rest_pos else n for i, degree in enumerate(degrees)])

            # 从图中删除顶点u以及与其相连的边以及其邻居
            # 先找邻居，然后删除和邻居连接的边
            max_violation_degree = 0.
            max_violation_edge = None

            delete_pos = set()
            delete_edge = set()
            for i, vio in enumerate(violation):
                if i in rest_edge:
                    if not vio[0].alpha[min_degree_pos] == 0:  # 当前边与顶点u连接
                        delete_edge.add(i)
                        if vio[1] > max_violation_degree:
                            max_violation_degree = vio[1]
                            max_violation_edge = vio[0]
                        for j in range(n):
                            if (j in rest_edge) and (not vio[0].alpha[j] == 0) and (not j == min_degree_pos):
                                delete_pos.add(j)
            # 更新剩余的边和顶点
            rest_edge = rest_edge - delete_edge
            rest_pos = rest_pos - delete_pos
            # 再删除与邻居相连的剩余的边
            for i, vio in enumerate(violation):
                if i in rest_edge:
                    for j in delete_pos:
                        if not vio[0].alpha[j] == 0:  # 当前边与邻居连接
                            delete_edge.add(i)
                            break
            # 再次更新
            rest_edge = rest_edge - delete_edge
            rest_pos = rest_pos - {min_degree_pos}

            # 最后删除顶点
            key_cell_pos.add(min_degree_pos)

            if max_violation_edge is not None:
                repair_edge.append(max_violation_edge)

    return repair_edge, key_cell_pos


def fd_detect(mts):
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本

    modified = modified.iloc[:, :mts.rfd_m]
    is_modified = is_modified.iloc[:, :mts.rfd_m]

    # 由于FD挖掘时无法识别浮点数，需要将数据修改为整形
    for col in modified.columns:
        # 乘100后四舍五入
        modified[col] = modified[col] * 100
        modified[col] = modified[col].astype(int)

    # 开始检测
    start = time.perf_counter()

    for lhs, rhs in mts.fds:
        for i in range(mts.len):
            for j in range(i + 1, mts.len):
                t_i = modified.iloc[i]
                t_j = modified.iloc[j]
                # 判断两个元组的条件
                condition = True
                for col, idx in lhs:
                    if not t_i[col] == t_j[col]:
                        condition = False
                        break
                if condition:
                    for col, idx in rhs:
                        if not t_i[col] == t_j[col]:
                            # print('检测到元组{}和元组{}违反FD: {} -> {}'.format(i, j, [col_ for col_, idx in lhs], col))
                            is_modified[col].values[i] = True
                            is_modified[col].values[j] = True
                            for col_, idx_ in lhs:
                                is_modified[col_].values[i] = True
                                is_modified[col_].values[j] = True

    time_cost += time.perf_counter() - start

    return modified, is_modified, time_cost


def rfd_detect(mts, method='is_cover'):
    modified = mts.modified.copy(deep=True)  # 拷贝数据
    is_modified = mts.isModified.copy(deep=True)  # 拷贝修复单元格信息
    time_cost = 0.  # 时间成本

    modified = modified.iloc[:, :mts.rfd_m]
    is_modified = is_modified.iloc[:, :mts.rfd_m]

    # 开始检测
    start = time.perf_counter()

    rfds = None
    if method == 'is_cover':
        rfds = mts.is_cover
    if method == 'domino':
        rfds = mts.domino
    if method == 'cords':
        rfds = mts.cords

    for conditions, y in rfds:
        # 先判断是否有条件，有条件才需要判断
        if len(conditions) > 0:
            for i in range(mts.len):
                for j in range(i + 1, mts.len):
                    # 获取元组
                    t_i = modified.iloc[i]
                    t_j = modified.iloc[j]
                    # 比较两个元组条件的差值
                    condition = True
                    for col, threshold in conditions:
                        if abs(t_i[col] - t_j[col]) > threshold:
                            condition = False
                            break
                    if condition and abs(t_i[y[0]] - t_j[y[0]]) > y[1]:
                        is_modified[y[0]].values[i] = True
                        is_modified[y[0]].values[j] = True
                        for col_, threshold_ in conditions:
                            is_modified[col_].values[i] = True
                            is_modified[col_].values[j] = True

    time_cost += time.perf_counter() - start

    return modified, is_modified, time_cost
