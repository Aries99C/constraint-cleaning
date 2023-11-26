from constraints.rfd.relax import FD
from constraints.rfd.relax import RFD
from constraints.rfd.pre import pre_glass
import numpy as np
import random

'''
 判断是否全为1，用于集合覆盖过程，判断是否所有的数据都被覆盖
 @param S 集合，为一个01组成的数组
 @return bool True:全为1  False：不全为1
'''


def all_one(S):
    for i in S:
        if i == 0:
            return False
    return True


'''
 针对01串进行或操作
 @param x,y 两个01串，为01组成的数组
 @return z 执行或操作后的结果，为01数组
'''


def my_or(x, y):
    z = []
    for i in range(len(x)):
        if x[i] == 1 or y[i] == 1:
            z.append(1)
        else:
            z.append(0)
    return z


'''
 判断两个01串是否存在包含关系，即是否x覆盖的数据y都覆盖，或这反之成立
 @param x,y 两个01串，为01组成的数组
 @return bool True:存在x覆盖y或者y覆盖x  False：不存在***
'''


def my_contain(x, y):
    flag = 1
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 1:
            flag = 0
            break
    if flag:
        return True
    flag = 1
    for i in range(len(y)):
        if y[i] == 0 and x[i] == 1:
            flag = 0
            break
    if flag:
        return True
    return False


'''
 根据对应的集合覆盖求得所有的极小覆盖集合（即LHS候选集合）
 @param op k*n k个长度为n个覆盖
 @return res 极小覆盖集合
'''


def orderedRelation(diff_list, i):
    return diff_list[np.argsort(diff_list[:, i])]
def fd_ok(op):
    ok = [[False] * len(op)] * len(op)  # ok[i][j] 表示i 和 j两个能否同时出现
    for i in range(len(op)):
        for j in range(i + 1, len(op)):
            if my_contain(op[i], op[j]):
                ok[i][j] = ok[j][i] = True
    res = []
    for i in range(len(op)):
        res.append(0)
        for j in range(len(op)):
            if i == j or not ok[i][j]:
                res[i] = res[i] << 1 | 1
            else:
                res[i] = res[i] << 1
    return res


'''
 根据对应的覆盖集合，生成后k条覆盖集合的或
 @param op，RHS op：k*n k个长度为n个覆盖  RHS：跳过的RHS属性
 @return res 后k条覆盖集合集
'''


def get_full_or(op, RHS):
    res = [[0] * len(op[0])] * (len(op) + 1)
    for i in range(len(op) - 1, -1, -1):
        if i == RHS:
            res[i] = res[i + 1]
            continue
        res[i] = my_or(res[i + 1], op[i])
    return res


def my_ok(x, y):
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 0:
            return False
    return True


def my_calc_or(x, y):
    res = []
    flag = 0
    for i in range(len(x)):
        if x[i] == 0 and y[i] == 1:
            flag = 1
        res.append(x[i] | y[i])
    if flag == 0:
        res[0] = -1
    return res


def my_all_one(S):
    for i in S:
        if i == 0:
            return False
    return True


def fd_LHS(op, RHS):  # 采用BFS的方法挑选LHS候选
    ok = fd_ok(op)
    queue = [[[0] * len(op[0]), 0, 0]]  # 第一个表示目前已经覆盖的行，第二表示目前已经挑选的列,第三个表示到了第几个列
    full_or = get_full_or(op, RHS)
    head = tail = 0
    ans = []
    while head <= tail:
        e = queue[head]
        al_cover_row = e[0]
        al_cover_col = e[1]
        al_num = e[2]
        head += 1
        if my_all_one(al_cover_row):
            ans.append(al_cover_col)
            continue
        if al_num >= len(op):
            continue
        if not my_ok(full_or[al_num], al_cover_row):
            continue
        for i in range(al_num, len(op)):
            if i == RHS:
                continue
            if ok[i] & al_cover_col:
                continue
            cnt = my_calc_or(al_cover_row, op[i])
            if cnt[0] == -1:
                continue
            tail += 1
            queue.append([cnt, al_cover_col | (1 << i), i + 1])

    return ans


def reduce(li):  # 二进制转化为数组
    new_li = list(set(li))
    res = []
    for i in new_li:
        res.append(bit_to_list(i))
    return res

def dominate(t_beta, diff_list_a):
    if len(t_beta)==0:
        return False
    cnt=1
    for h in t_beta:
        for j in range(len(h)):
            cnt+=1
            if h[j] > diff_list_a[j]:
                return False
            if cnt==1000:
                return True
    return True

def gen_LHS(beta_1,beta_2,RHS):

    min_value = []
    import time
    Start=time.time()
    for j in range(len(beta_1[0])):
        maxn = 0
        for i in range(len(beta_1)):
            maxn = max(maxn, beta_1[i][j])
        min_value.append(maxn)
    End=time.time()
    op = [[] for i in range(len(beta_1[0]))]
    for i in beta_2:
        for j in range(len(i)):
            if i[j] < min_value[j]:
                op[j].append(0)
            else:
                op[j].append(1)
    res = fd_LHS(op, RHS)
    res = reduce(res)
    return res


def orderedRelation(diff_list, i):
    return diff_list[np.argsort(diff_list[:, i])]


def bit_to_list(t):
    S = []
    cnt = -1
    while t:
        cnt += 1
        op = t % 2
        t = t >> 1
        if op == 1:
            S.append(cnt)
    return S

def generte(data):
    res = []
    data_new=[]
    for i in data:
        if dominate(data_new, i):
            continue
        data_new.append(i)
    data=np.array(data_new)
    for i in range(len(data[0])):
        diff_list_a = orderedRelation(data, i)
        df = [data[ll][i] for ll in range(len(data))]
        df = list(set(df))
        df.sort()
        p = len(df) - 1
        step = 1
        beta1=diff_list_a
        beta2=[]
        while p >= 0:
            k = df[p]
            while len(beta1)!=0 and beta1[len(beta1)-1][i]>df[p]:
                x=beta1[len(beta1)-1]
                beta1=beta1[:len(beta1)-1]
                beta2.append(x)

            LHS_list = gen_LHS(beta1,beta2, i)
            for j in LHS_list:
                now_FD = FD(j, i)
                new_RFD = RFD(now_FD, k)
                new_RFD.generator(data)
                res.append(new_RFD)
            p = p - step
            step *= 2
    return res


def mine(distance, mts):
    # 挖掘RFD
    res = generte(distance)
    # 存储RFD
    rfds = []
    for i in res:
        # 条件X，可能为空
        conditions = []
        for j in i.LHS:
            conditions.append((mts.cols[j[0]], j[1]))  # 每个条件X的形式为：(属性, 属性在元组间允许的差值范围)
        # 映射Y
        y = (mts.cols[i.RHS[0]], i.RHS[1])

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


def make_data(Distance_new, rate):
    for i in range(len(Distance_new)):
        rad = random.random()
        if rad < rate:
            Distance_new[i] = Distance_new[i] * rate
    return Distance_new


def is_cover_mining_rfd(mts, n=100, m=10, verbose=0):
    # 读取部分数据
    distance, name_list = pre_glass(mts, n, m)
    # 挖掘规则
    rfds = mine(distance, mts)

    if verbose > 0:
        print('{:=^80}'.format(' 在数据集{}上挖掘松弛函数依赖-CORDS '.format(mts.dataset.upper())))
        for conditions, y in rfds:
            if len(conditions) == 0:
                print('松弛函数依赖: d({}) <= {:.3f}'.format(y[0], y[1]))
            else:
                print('松弛函数依赖: {} --> {}'.format(['d({}) <= {:.3f}'.format(cond[0], cond[1]) for cond in conditions], 'd({}) <= {:.3f}'.format(y[0], y[1])))

    return rfds
