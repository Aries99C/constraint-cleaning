from utils import project_root

PROJECT_ROOT = project_root()


def read_from_TANE(mts, verbose=0):
    fd_list = []
    with open(PROJECT_ROOT + '/constraints/rules/{}_fd_raw.txt'.format(mts.dataset), 'r') as f:
        for line in f.readlines():
            lhs_str, rhs_str = line.split(',')[0].strip(), line.split(',')[1].strip()
            lhs = []
            for x in lhs_str.split(' '):
                x_idx = int(x) - 1  # TANE挖掘到变量X的列索引
                x_name = mts.cols[x_idx]
                lhs.append((x_name, x_idx))
            rhs = (mts.cols[int(rhs_str) - 1], int(rhs_str) - 1)  # TANE挖掘到的变量Y的列索引
            duplicate = False
            for cmp_lhs, cmp_rhs in fd_list:
                duplicate_cnt = 0
                for cmp_col, cmp_idx in cmp_lhs:
                    for col, idx in lhs:
                        if cmp_col == col:
                            duplicate_cnt += 1
                if duplicate_cnt == len(cmp_lhs):
                    duplicate = True
                    cmp_rhs.append(rhs)
                    break
            if not duplicate:
                fd_list.append((lhs, [rhs]))
    if verbose > 0:
        print('{:=^80}'.format(' 在数据集{}上挖掘函数依赖 '.format(mts.dataset.upper())))
        for lhs, rhs in fd_list:
            print('函数依赖: {} -> {}'.format([var[0] for var in lhs], [var[0] for var in rhs]))
    return fd_list
