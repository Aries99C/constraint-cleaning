import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from cleaning.benchmark import delta, raa, speed_local, speed_global, acc_local, acc_global, IMR, ewma, median_filter, \
    func_lp
from dataset.mts import MTS
from utils import project_root


PROJECT_ROOT = project_root()


def f1(is_modified, is_dirty):
    a = is_modified.values.reshape(-1)
    b = is_dirty.values.reshape(-1)

    p, r, f = precision_score(a, b), recall_score(a, b), f1_score(a, b)

    return p, r, f


def benchmark_performance(dataset='idf', index_col='timestamp', datetime_index=True, w=3,
                          lens=range(2000, 10000, 2000), ratios=np.arange(0.05, 0.25, 0.05),
                          constraints=None):
    if constraints is None:
        constraints = ['speed', 'acc']

    # 数据集长度的对比实验
    len_error_performance = pd.DataFrame(
        columns=['data_len', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    len_time_performance = pd.DataFrame(
        columns=['data_len', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    len_raa_performance = pd.DataFrame(
        columns=['data_len', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    len_precision_performance = pd.DataFrame(
        columns=['data_len', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    len_recall_performance = pd.DataFrame(
        columns=['data_len', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    len_f1_performance = pd.DataFrame(
        columns=['data_len', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )

    for data_len in lens:
        mts = MTS(dataset, index_col, datetime_index, data_len, verbose=1)
        mts.constraints_mining(pre_mined=True, mining_constraints=constraints, w=w, verbose=1)
        mts.insert_error(ratio=0.1, snr=15, verbose=1)

        # 速度约束Local修复
        speed_local_modified, speed_local_is_modified, speed_local_time = speed_local(mts)
        print('{:=^80}'.format(' 局部速度约束修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(speed_local_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(speed_local_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, speed_local_modified)))

        # 速度约束Global修复
        speed_global_modified, speed_global_is_modified, speed_global_time = speed_global(mts)
        print('{:=^80}'.format(' 全局速度约束修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(speed_global_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(speed_global_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, speed_global_modified)))

        # 速度约束+加速度约束Local修复
        acc_local_modified, acc_local_is_modified, acc_local_time = acc_local(mts)
        print('{:=^80}'.format(' 局部速度约束+加速度约束修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(acc_local_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(acc_local_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, acc_local_modified)))

        # 速度约束+加速度约束Global修复
        acc_global_modified, acc_global_is_modified, acc_global_time = acc_global(mts)
        print('{:=^80}'.format(' 全局速度约束+加速度约束修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(acc_global_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(acc_global_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, acc_global_modified)))

        # IMR修复
        imr_modified, imr_is_modified, imr_time = IMR(max_iter=1000).clean(mts)
        print('{:=^80}'.format(' IMR算法修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(imr_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(imr_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, imr_modified)))

        # EWMA修复
        ewma_modified, ewma_is_modified, ewma_time = ewma(mts, beta=0.9)
        print('{:=^80}'.format(' EWMA算法修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(ewma_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(ewma_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, ewma_modified)))

        # 中值滤波器修复
        median_filter_modified, median_filter_is_modified, median_filter_time = median_filter(mts, w=10)
        print('{:=^80}'.format(' 中值滤波修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(median_filter_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(median_filter_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, median_filter_modified)))

        # 记录实验结果
        len_error_performance.loc[len(len_error_performance)] = [
            data_len,
            delta(speed_local_modified, mts.clean),
            delta(speed_global_modified, mts.clean),
            delta(acc_local_modified, mts.clean),
            delta(acc_global_modified, mts.clean),
            delta(imr_modified, mts.clean),
            delta(ewma_modified, mts.clean),
            delta(median_filter_modified, mts.clean)
        ]
        len_time_performance.loc[len(len_time_performance)] = [
            data_len,
            speed_local_time,
            speed_global_time,
            acc_local_time,
            acc_global_time,
            imr_time,
            ewma_time,
            median_filter_time
        ]
        len_raa_performance.loc[len(len_raa_performance)] = [
            data_len,
            raa(mts.origin, mts.clean, speed_local_modified),
            raa(mts.origin, mts.clean, speed_global_modified),
            raa(mts.origin, mts.clean, acc_local_modified),
            raa(mts.origin, mts.clean, acc_global_modified),
            raa(mts.origin, mts.clean, imr_modified),
            raa(mts.origin, mts.clean, ewma_modified),
            raa(mts.origin, mts.clean, median_filter_modified)
        ]
        len_precision_performance.loc[len(len_precision_performance)] = [
            data_len,
            f1(speed_local_is_modified, mts.isDirty)[0],
            f1(speed_global_is_modified, mts.isDirty)[0],
            f1(acc_local_is_modified, mts.isDirty)[0],
            f1(acc_global_is_modified, mts.isDirty)[0],
            f1(imr_is_modified, mts.isDirty)[0],
            f1(ewma_is_modified, mts.isDirty)[0],
            f1(median_filter_is_modified, mts.isDirty)[0]
        ]
        len_recall_performance.loc[len(len_recall_performance)] = [
            data_len,
            f1(speed_local_is_modified, mts.isDirty)[1],
            f1(speed_global_is_modified, mts.isDirty)[1],
            f1(acc_local_is_modified, mts.isDirty)[1],
            f1(acc_global_is_modified, mts.isDirty)[1],
            f1(imr_is_modified, mts.isDirty)[1],
            f1(ewma_is_modified, mts.isDirty)[1],
            f1(median_filter_is_modified, mts.isDirty)[1]
        ]
        len_f1_performance.loc[len(len_f1_performance)] = [
            data_len,
            f1(speed_local_is_modified, mts.isDirty)[2],
            f1(speed_global_is_modified, mts.isDirty)[2],
            f1(acc_local_is_modified, mts.isDirty)[2],
            f1(acc_global_is_modified, mts.isDirty)[2],
            f1(imr_is_modified, mts.isDirty)[2],
            f1(ewma_is_modified, mts.isDirty)[2],
            f1(median_filter_is_modified, mts.isDirty)[2]
        ]

    # 数据集错误比例的对比实验
    ratio_error_performance = pd.DataFrame(
        columns=['error_ratio', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    ratio_time_performance = pd.DataFrame(
        columns=['error_ratio', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    ratio_raa_performance = pd.DataFrame(
        columns=['error_ratio', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    ratio_precision_performance = pd.DataFrame(
        columns=['error_ratio', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    ratio_recall_performance = pd.DataFrame(
        columns=['error_ratio', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )
    ratio_f1_performance = pd.DataFrame(
        columns=['error_ratio', 'Speed(L)', 'Speed(G)', 'speed+Acc(L)', 'speed+Acc(G)', 'IMR', 'EWMA', 'Median']
    )

    for error_ratio in ratios:
        mts = MTS(dataset, index_col, datetime_index, 5000, verbose=1)
        mts.constraints_mining(pre_mined=True, mining_constraints=constraints, w=w, verbose=1)
        mts.insert_error(ratio=error_ratio, snr=15, verbose=1)

        # 速度约束Local修复
        speed_local_modified, speed_local_is_modified, speed_local_time = speed_local(mts)
        print('{:=^80}'.format(' 局部速度约束修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(speed_local_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(speed_local_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, speed_local_modified)))

        # 速度约束Global修复
        speed_global_modified, speed_global_is_modified, speed_global_time = speed_global(mts)
        print('{:=^80}'.format(' 全局速度约束修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(speed_global_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(speed_global_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, speed_global_modified)))

        # 速度约束+加速度约束Local修复
        acc_local_modified, acc_local_is_modified, acc_local_time = acc_local(mts)
        print('{:=^80}'.format(' 局部速度约束+加速度约束修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(acc_local_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(acc_local_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, acc_local_modified)))

        # 速度约束+加速度约束Global修复
        acc_global_modified, acc_global_is_modified, acc_global_time = acc_global(mts)
        print('{:=^80}'.format(' 全局速度约束+加速度约束修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(acc_global_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(acc_global_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, acc_global_modified)))

        # IMR修复
        imr_modified, imr_is_modified, imr_time = IMR(max_iter=1000).clean(mts)
        print('{:=^80}'.format(' IMR算法修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(imr_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(imr_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, imr_modified)))

        # EWMA修复
        ewma_modified, ewma_is_modified, ewma_time = ewma(mts, beta=0.9)
        print('{:=^80}'.format(' EWMA算法修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(ewma_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(ewma_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, ewma_modified)))

        # 中值滤波器修复
        median_filter_modified, median_filter_is_modified, median_filter_time = median_filter(mts, w=10)
        print('{:=^80}'.format(' 中值滤波修复数据集{} '.format(mts.dataset.upper())))
        print('修复用时: {:.4g}ms'.format(median_filter_time))
        print('修复值与正确值平均误差: {:.4g}'.format(delta(median_filter_modified, mts.clean)))
        print('修复相对精度: {:.4g}'.format(raa(mts.origin, mts.clean, median_filter_modified)))

        # 记录实验结果
        ratio_error_performance.loc[len(ratio_error_performance)] = [
            error_ratio,
            delta(speed_local_modified, mts.clean),
            delta(speed_global_modified, mts.clean),
            delta(acc_local_modified, mts.clean),
            delta(acc_global_modified, mts.clean),
            delta(imr_modified, mts.clean),
            delta(ewma_modified, mts.clean),
            delta(median_filter_modified, mts.clean)
        ]
        ratio_time_performance.loc[len(ratio_time_performance)] = [
            error_ratio,
            speed_local_time,
            speed_global_time,
            acc_local_time,
            acc_global_time,
            imr_time,
            ewma_time,
            median_filter_time
        ]
        ratio_raa_performance.loc[len(ratio_raa_performance)] = [
            error_ratio,
            raa(mts.origin, mts.clean, speed_local_modified),
            raa(mts.origin, mts.clean, speed_global_modified),
            raa(mts.origin, mts.clean, acc_local_modified),
            raa(mts.origin, mts.clean, acc_global_modified),
            raa(mts.origin, mts.clean, imr_modified),
            raa(mts.origin, mts.clean, ewma_modified),
            raa(mts.origin, mts.clean, median_filter_modified)
        ]
        ratio_precision_performance.loc[len(ratio_precision_performance)] = [
            error_ratio,
            f1(speed_local_is_modified, mts.isDirty)[0],
            f1(speed_global_is_modified, mts.isDirty)[0],
            f1(acc_local_is_modified, mts.isDirty)[0],
            f1(acc_global_is_modified, mts.isDirty)[0],
            f1(imr_is_modified, mts.isDirty)[0],
            f1(ewma_is_modified, mts.isDirty)[0],
            f1(median_filter_is_modified, mts.isDirty)[0]
        ]
        ratio_recall_performance.loc[len(ratio_recall_performance)] = [
            error_ratio,
            f1(speed_local_is_modified, mts.isDirty)[1],
            f1(speed_global_is_modified, mts.isDirty)[1],
            f1(acc_local_is_modified, mts.isDirty)[1],
            f1(acc_global_is_modified, mts.isDirty)[1],
            f1(imr_is_modified, mts.isDirty)[1],
            f1(ewma_is_modified, mts.isDirty)[1],
            f1(median_filter_is_modified, mts.isDirty)[1]
        ]
        ratio_f1_performance.loc[len(ratio_f1_performance)] = [
            error_ratio,
            f1(speed_local_is_modified, mts.isDirty)[2],
            f1(speed_global_is_modified, mts.isDirty)[2],
            f1(acc_local_is_modified, mts.isDirty)[2],
            f1(acc_global_is_modified, mts.isDirty)[2],
            f1(imr_is_modified, mts.isDirty)[2],
            f1(ewma_is_modified, mts.isDirty)[2],
            f1(median_filter_is_modified, mts.isDirty)[2]
        ]

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
    benchmark_performance(lens=range(2000, 20000+1, 2000), ratios=np.arange(0.05, 0.35+0.01, 0.05))
