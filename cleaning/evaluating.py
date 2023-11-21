import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
from cleaning.benchmark import delta, raa, speed_local, speed_global, acc_local, acc_global, IMR, ewma, median_filter, \
    func_lp
from dataset.mts import MTS
from utils import project_root

warnings.filterwarnings('ignore')

PROJECT_ROOT = project_root()
func_dict = {
    'Speed(L)': speed_local,
    'Speed(G)': speed_global,
    'Speed+Acc(L)': acc_local,
    'Speed+Acc(G)': acc_global,
    'IMR': IMR().clean,
    'EWMA': ewma,
    'Median': median_filter,
    'Func_LP': func_lp,
}


def f1(is_modified, is_dirty):
    a = is_modified.values.reshape(-1)
    b = is_dirty.values.reshape(-1)

    p, r, f = precision_score(a, b), recall_score(a, b), f1_score(a, b)

    return p, r, f


def benchmark_performance(dataset='idf', index_col='timestamp', datetime_index=True, w=3,
                          lens=range(2000, 10000, 2000), ratios=np.arange(0.05, 0.25, 0.05),
                          constraints=None, methods=None):
    if methods is None:
        methods = ['Speed(L)', 'Speed(G)', 'Speed+Acc(L)', 'Speed+Acc(G)', 'IMR', 'EWMA', 'Median', 'Func_LP']
    if constraints is None:
        constraints = ['speed', 'acc', 'stcd']

    # 数据集长度的对比实验
    len_error_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    len_time_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    len_raa_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    len_precision_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    len_recall_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    len_f1_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )

    idx = 0

    for data_len in lens:
        mts = MTS(dataset, index_col, datetime_index, data_len, verbose=1)
        mts.constraints_mining(pre_mined=True, mining_constraints=constraints, w=w, verbose=1)
        mts.insert_error(ratio=0.1, snr=15, verbose=1)

        errors = []
        raas = []
        times = []
        ps = []
        rs = []
        fs = []

        # 对测试列表中的所有方法都执行测试
        for method in methods:
            if method in ['Speed(L)', 'Speed+Acc(L)', 'Median', 'Func_LP']:
                modified, is_modified, time = func_dict[method](mts, w=w)
            elif method in ['Speed(G)', 'Speed+Acc(G)']:
                modified, is_modified, time = func_dict[method](mts, w=w, x=5)
            else:
                modified, is_modified, time = func_dict[method](mts)

            error = delta(modified, mts.clean) / mts.delta_clean_origin()
            raa_score = raa(mts.origin, mts.clean, modified)
            p, r, f = f1(is_modified, mts.isDirty)

            errors.append(error)
            raas.append(raa_score)
            times.append(time)
            ps.append(p)
            rs.append(r)
            fs.append(f)

            print('{:=^80}'.format(' {}修复数据集{} '.format(method, mts.dataset.upper())))
            print('修复用时: {:.4g}ms'.format(time))
            print('修复误差比: {:.4g}'.format(error))
            print('修复相对精度: {:.4g}'.format(raa_score))

        # 记录实验结果
        len_error_performance.loc[idx] = [data_len] + errors
        len_raa_performance.loc[idx] = [data_len] + raas
        len_time_performance.loc[idx] = [data_len] + times
        len_precision_performance.loc[idx] = [data_len] + ps
        len_recall_performance.loc[idx] = [data_len] + rs
        len_f1_performance.loc[idx] = [data_len] + fs

        idx += 1

    # 数据集错误比例的对比实验
    ratio_error_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    ratio_time_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    ratio_raa_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    ratio_precision_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    ratio_recall_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )
    ratio_f1_performance = pd.DataFrame(
        columns=['data_len'] + [func_name for func_name in methods]
    )

    idx = 0

    for error_ratio in ratios:
        mts = MTS(dataset, index_col, datetime_index, 5000, verbose=1)
        mts.constraints_mining(pre_mined=True, mining_constraints=constraints, w=w, verbose=1)
        mts.insert_error(ratio=error_ratio, snr=15, verbose=1)

        errors = []
        raas = []
        times = []
        ps = []
        rs = []
        fs = []

        # 对测试列表中的所有方法都执行测试
        for method in methods:
            if method in ['Speed(L)', 'speed+Acc(L)', 'Median', 'Func_LP']:
                modified, is_modified, time = func_dict[method](mts, w=w)
            elif method in ['Speed(G)', 'speed+Acc(G)']:
                modified, is_modified, time = func_dict[method](mts, w=w, x=5)
            else:
                modified, is_modified, time = func_dict[method](mts)

            error = delta(modified, mts.clean) / mts.delta_clean_origin()
            raa_score = raa(mts.origin, mts.clean, modified)
            p, r, f = f1(is_modified, mts.isDirty)

            errors.append(error)
            raas.append(raa_score)
            times.append(time)
            ps.append(p)
            rs.append(r)
            fs.append(f)

            print('{:=^80}'.format(' {}修复数据集{} '.format(method, mts.dataset.upper())))
            print('修复用时: {:.4g}ms'.format(time))
            print('修复误差比: {:.4g}'.format(error))
            print('修复相对精度: {:.4g}'.format(raa_score))

        # 记录实验结果
        ratio_error_performance.loc[idx] = [error_ratio] + errors
        ratio_raa_performance.loc[idx] = [error_ratio] + raas
        ratio_time_performance.loc[idx] = [error_ratio] + times
        ratio_precision_performance.loc[idx] = [error_ratio] + ps
        ratio_recall_performance.loc[idx] = [error_ratio] + rs
        ratio_f1_performance.loc[idx] = [error_ratio] + fs

        idx += 1

    # 保存数据
    len_error_performance.to_csv(PROJECT_ROOT + '/{}_len_error.csv'.format(dataset.upper()))
    len_time_performance.to_csv(PROJECT_ROOT + '/{}_len_time.csv'.format(dataset.upper()))
    len_raa_performance.to_csv(PROJECT_ROOT + '/{}_len_raa.csv'.format(dataset.upper()))
    len_precision_performance.to_csv(PROJECT_ROOT + '/{}_len_precision.csv'.format(dataset.upper()))
    len_recall_performance.to_csv(PROJECT_ROOT + '/{}_len_recall.csv'.format(dataset.upper()))
    len_f1_performance.to_csv(PROJECT_ROOT + '/{}_len_f1.csv'.format(dataset.upper()))

    ratio_error_performance.to_csv(PROJECT_ROOT + '/{}_ratio_error.csv'.format(dataset.upper()))
    ratio_time_performance.to_csv(PROJECT_ROOT + '/{}_ratio_time.csv'.format(dataset.upper()))
    ratio_raa_performance.to_csv(PROJECT_ROOT + '/{}_ratio_raa.csv'.format(dataset.upper()))
    ratio_precision_performance.to_csv(PROJECT_ROOT + '/{}_ratio_precision.csv'.format(dataset.upper()))
    ratio_recall_performance.to_csv(PROJECT_ROOT + '/{}_ratio_recall.csv'.format(dataset.upper()))
    ratio_f1_performance.to_csv(PROJECT_ROOT + '/{}_ratio_f1.csv'.format(dataset.upper()))


if __name__ == '__main__':
    benchmark_performance(lens=range(2000, 20000 + 1, 2000), ratios=np.arange(0.05, 0.35 + 0.01, 0.05))
