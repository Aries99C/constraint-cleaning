import numpy as np
import random

from constraints.rfd.pre import pre_glass


def getDistanceRelation(r):
    res = []
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            kk = []
            for k in range(len(r[i])):
                kk.append(abs(r[i][k] - r[j][k]))
            res.append(kk)
    res = np.array(res)
    return res


def orderedRelation(diff_list, i):
    return diff_list[np.argsort(diff_list[:, i])]


def dominate(t_beta, diff_list_a):
    for h in t_beta:
        flag = True
        for j in range(len(h)):
            if h[j] > diff_list_a[j]:
                flag = False
        if flag:
            return True
    return False


def getTRel(diff_list_a, i):
    t = []
    t_beta = []
    w = len(diff_list_a)
    step = 1
    while w > 0:
        beta = max(w - step, 0)
        while beta != 0 and diff_list_a[beta - 1][i] == diff_list_a[beta][i]:
            beta = beta - 1
        for j in range(beta, w):
            if dominate(t_beta, diff_list_a[j]):
                continue
            t_beta.append(diff_list_a[j])
        t.append((t_beta, (i, diff_list_a[beta][i])))
        w = beta
        step = step * 2
    return t


def viable(x, dm):
    for h in dm:
        flag1 = False
        flag2 = True
        for f in x:
            if h[f] != 0:
                flag2 = False
            if h[f] < 0:
                flag1 = True
        if (not flag1) and (not flag2):
            return False
    return True


def redundant(x_new, lhs):
    for i in lhs:
        d = [False for c in i not in x_new]
        if not d:
            return True
    return False


def generateRFDcs(t_beta, w, lhs):
    rfd_cs = []
    for cc in lhs:
        lhs_now = []
        for i in cc:
            if i == cc[0]:
                lhs_now.append([i, w[i] - 0.001])
                continue
            lhs_now.append([i, w[i]])
        for j in range(1, len(cc)):
            kk = 99999999
            for i in t_beta:
                if i == w:
                    continue
                if t_beta[i][cc[j]] <= w[cc[j]]:
                    continue
                if t_beta[i][cc[0]] >= w[cc[0]]:
                    continue
                flag = True
                for e in range(j):
                    if t_beta[i][cc[e]] > lhs_now[e][1]:
                        flag = False
                if not flag:
                    continue
                flag = False
                for e in range(j + 1, len(cc)):
                    if t_beta[i][cc[e]] <= lhs_now[e][1]:
                        flag = True
                if not flag:
                    continue
                kk = min(kk, t_beta[i][cc[j]])
            if kk == 99999999:
                continue
            lhs_now[j][1] = kk - 0.001
        rfd_cs.append(lhs_now)
    return rfd_cs


def getRFDs(t_beta, a_i, w):
    L = 1
    levelAttr = []
    LHS = []
    DM = []
    for i in t_beta:
        if i.all() == w.all():
            continue
        kk = []
        for j in range(len(w)):
            kk.append(i[j] - w[j])
        DM.append(kk)
    for i in range(len(w)):
        if i == a_i:
            continue
        levelAttr.append([i])
    while L <= len(w) and len(levelAttr) != 0:
        L_nxt = []
        for X in levelAttr:
            if viable(X, DM):
                LHS.append(X)
        levelAttr = []
        for X in levelAttr:
            if not viable(X, DM):
                for kk in range(len(w)):
                    if kk == a_i:
                        continue
                    X_new = X
                    X_new.append(kk)
                    if not redundant(X_new, LHS):
                        levelAttr.append(X_new)
        L = L + 1
    return generateRFDcs(t_beta, w, LHS)


def Domino(r):
    RFDcs = []
    diff_list = r
    cnt = 1
    for i in range(len(diff_list[0])):
        diff_list_a = orderedRelation(diff_list, i)
        T = getTRel(diff_list_a, i)
        for T_beta in T:
            RHS = T_beta[1]
            T_beta = T_beta[0]
            for w in T_beta:
                LHS_now = getRFDs(T_beta, i, w)
                for kk in LHS_now:
                    cnt += 1
                    if cnt & 10 == 10:
                        RFDcs.append((kk, RHS))
    return RFDcs


def make_data(distance_new, rate):
    for i in range(len(distance_new)):
        rad = random.random()
        if rad < rate:
            distance_new[i] = distance_new[i] * rate
    return distance_new


def mine(distance, mts):
    # 挖掘RFD
    res = Domino(distance)
    # 存储RFD
    rfds = []
    for i in res:
        # 条件X，可能为空
        conditions = []
        for j in i[0]:
            conditions.append((mts.cols[j[0]], j[1]))  # 每个条件X的形式为：(属性, 属性在元组间允许的差值范围)
        # 映射Y
        y = (mts.cols[i[1][0]], i[1][1])
        rfds.append((conditions, y))

        # 判断RFD重复
        duplicate = False
        for idx, (cmp_conditions, cmp_y) in enumerate(rfds):
            if len(cmp_conditions) == len(conditions):  # 可能存在X重复
                x_duplicate = True
                for k in range(len(conditions)):
                    if not cmp_conditions[k][0] == conditions[k][0]:
                        x_duplicate = False
                        break
                if x_duplicate and cmp_y[0] == y[0]:  # 确定X和Y重复
                    new_conditions = []
                    for k in range(len(conditions)):
                        new_conditions.append((conditions[k][0], max(cmp_conditions[k][1], conditions[k][1])))
                    new_y = (cmp_y[0], max(cmp_y[1], y[1]))
                    rfds[idx] = (new_conditions, new_y)
                    duplicate = True
                    break
        if not duplicate:
            rfds.append((conditions, y))

    return rfds


def domino_mining_rfd(mts, n=100, m=10, verbose=0):
    # 读取部分数据
    distance, name_list = pre_glass(mts, n, m)
    # 挖掘规则
    rfds = mine(distance, mts)

    if verbose > 0:
        print('{:=^80}'.format(' 在数据集{}上挖掘松弛函数依赖-Domino '.format(mts.dataset.upper())))
        for conditions, y in rfds:
            if len(conditions) == 0:
                print('松弛函数依赖: d({}) <= {:.3f}'.format(y[0], y[1]))
            else:
                print('松弛函数依赖: {} --> {}'.format(['d({}) <= {:.3f}'.format(cond[0], cond[1]) for cond in conditions], 'd({}) <= {:.3f}'.format(y[0], y[1])))

    return rfds
