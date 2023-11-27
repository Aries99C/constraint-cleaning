import numpy as np
import pandas as pd
import warnings

from cleaning.benchmark import delta, raa, f1, speed_local, speed_global, acc_local, acc_global, IMR, ewma, \
    median_filter, \
    func_lp, func_mvc
from cleaning.benchmark import fd_detect, rfd_detect
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
    'Func_MVC': func_mvc,
    'fd': fd_detect,
    'domino': rfd_detect,
    'cords': rfd_detect,
    'is_cover': rfd_detect
}


def benchmark_performance(dataset='idf', index_col='timestamp', datetime_index=True, w=2,
                          lens=range(2000, 20000 + 1, 2000), ratios=np.arange(0.05, 0.35 + 0.01, 0.05),
                          constraints=None, methods=None):
    if methods is None:
        methods = ['Func_MVC', 'Func_LP', 'EWMA', 'Speed(G)', 'Speed+Acc(G)', 'Median', 'IMR', 'Speed(L)',
                   'Speed+Acc(L)']
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
            if method in ['Speed(L)', 'Speed+Acc(L)', 'Median', 'Func_LP', 'Func_MVC']:
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
            if method in ['Speed(L)', 'speed+Acc(L)', 'Median', 'Func_LP', 'Func_MVC']:
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
    len_error_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_len_error.csv'.format(dataset.upper(), dataset.upper()), index=False)
    len_time_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_len_time.csv'.format(dataset.upper(), dataset.upper()), index=False)
    len_raa_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_len_raa.csv'.format(dataset.upper(), dataset.upper()), index=False)
    len_precision_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_len_precision.csv'.format(dataset.upper(), dataset.upper()), index=False)
    len_recall_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_len_recall.csv'.format(dataset.upper(), dataset.upper()), index=False)
    len_f1_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_len_f1.csv'.format(dataset.upper(), dataset.upper()), index=False)

    ratio_error_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_ratio_error.csv'.format(dataset.upper(), dataset.upper()), index=False)
    ratio_time_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_ratio_time.csv'.format(dataset.upper(), dataset.upper()), index=False)
    ratio_raa_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_ratio_raa.csv'.format(dataset.upper(), dataset.upper()), index=False)
    ratio_precision_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_ratio_precision.csv'.format(dataset.upper(), dataset.upper()), index=False)
    ratio_recall_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_ratio_recall.csv'.format(dataset.upper(), dataset.upper()), index=False)
    ratio_f1_performance.to_csv(PROJECT_ROOT + '/exp/{}/{}_ratio_f1.csv'.format(dataset.upper(), dataset.upper()), index=False)


def fd_and_rfd(dataset='idf', index_col='timestamp', datetime_index=True,
               lens=range(2000, 20000 + 1, 2000), methods=None):
    if methods is None:
        methods = ['fd', 'domino', 'cords', 'is_cover']

    len_time_performance = pd.DataFrame(
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
        mts.constraints_mining(pre_mined=True, mining_constraints=['fd', 'domino', 'cords', 'is_cover'], verbose=1)
        mts.insert_error(ratio=0.2, snr=15, verbose=1)

        times = []
        ps = []
        rs = []
        fs = []

        # 对测试列表中的所有方法都执行测试
        for method in methods:
            if method in ['domino', 'cords', 'is_cover']:
                modified, is_modified, time = func_dict[method](mts, method)
            else:
                modified, is_modified, time = func_dict[method](mts)

            p, r, f = f1(is_modified, mts.isDirty.iloc[:, :mts.rfd_m])

            times.append(time)
            ps.append(p)
            rs.append(r)
            fs.append(f)

            print('{:=^80}'.format(' {}检测数据集{} '.format(method, mts.dataset.upper())))
            print('检测用时: {:.4g}ms'.format(time))
            print('Precision: {:.4g}, Recall: {:.4g}, F1: {:.4g}'.format(p, r, f))

        # 记录实验结果
        len_time_performance.loc[idx] = [data_len] + times
        len_precision_performance.loc[idx] = [data_len] + ps
        len_recall_performance.loc[idx] = [data_len] + rs
        len_f1_performance.loc[idx] = [data_len] + fs

        idx += 1

        # 保存数据
        len_time_performance.to_csv(PROJECT_ROOT + '/detect_{}_len_time.csv'.format(dataset.upper()), index=False)
        len_precision_performance.to_csv(PROJECT_ROOT + '/detect_{}_len_precision.csv'.format(dataset.upper()), index=False)
        len_recall_performance.to_csv(PROJECT_ROOT + '/detect_{}_len_recall.csv'.format(dataset.upper()), index=False)
        len_f1_performance.to_csv(PROJECT_ROOT + '/detect_{}_len_f1.csv'.format(dataset.upper()), index=False)


if __name__ == '__main__':
    benchmark_performance(dataset='WADI', index_col='Row', datetime_index=False, lens=range(2000, 20000 + 1, 2000), ratios=np.arange(0.05, 0.35 + 0.01, 0.05))
    # fd_and_rfd(lens=range(100, 500 + 1, 100))
