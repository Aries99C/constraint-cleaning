import math
import time
import numpy as np

from scipy.optimize import linprog


def delta_modified_clean(modified, clean):
    df = clean.sub(modified)    # 两个mts相减
    x = np.absolute(df.values)  # 获取ndarray的绝对值
    return np.sum(x)


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
