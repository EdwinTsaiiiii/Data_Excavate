import numpy as np
import itertools

# 计算支持度
def compute_sup(S,D):
    n,m = D.shape
    sup = 0
    for i in range(0,n):
        if all(S <= D[i,:]):
            sup = sup + 1
    return sup/n

# 计算n项集
def c_set(k,D):
    n, m = D.shape
    result_i = [0]*m
    for i in range(0,k):
        result_i[i] = 1
    # 求排列组合
    res_list = list(itertools.permutations(result_i, m))
    # 去重
    res_c = np.array(list(set([tuple(t) for t in res_list])))
    return res_c

# 计算可信度
def compute_conf(S,D):
    n, m = D.shape
    set_1 = [0] * m
    set_2 = [0] * m
    for i in range(0,m):
        if S[i] == -1:
            set_2[i] = 1
        if S[i] == -1 or S[i] == 1:
            set_1[i] = 1
    confidence = compute_sup(set_1,data) / compute_sup(set_2,data)
    return confidence

# 计算项集的规则
def c_rule(S):
    num = np.sum(S == 1)
    basic_i = [1] * num
    basic = np.array([basic_i] * (num - 1))
    basic_res = []
    for i in range(0,num - 1):
        basic[i,0:i+1] = -1
        basic_all = list(itertools.permutations(basic[i,:], len(basic[i,:])))
        basic_real = list(set([tuple(t) for t in basic_all]))
        for j in basic_real:
            basic_res.append(j)
    result = []
    basic_res = np.array(basic_res)
    for i in range(0, len(basic_res)):
        S_copy = np.array(S)
        count = 0
        for j in range(0, len(S)):
            if S[j] == 1:
                S_copy[j] = basic_res[i,count]
                count = count + 1
        result.append(S_copy)
    return np.array(result)

def apriori(D,min_sup,min_conf):
    n,m = D.shape
    freq_set3 = []
    for n in range(1, m):
        C = c_set(n, D)
        n_C, m_C = C.shape
        freq_set = []
        # 求候选n项集
        print('----------------------------候选' + str(n) + '项集----------------------------')
        for i in range(0, n_C):
            sup = compute_sup(C[i, :], D)
            flag = sup >= min_sup
            if sup > 0:
                print('候选集:', C[i, :], '支持度：', sup, '是否为频繁项集：', flag)
            if flag == True:
                freq_set.append(C[i, :])
        print('----------------------------频繁' + str(n) + '项集----------------------------')
        print(np.array(freq_set))
        if n == 3:
            freq_set3 = np.array(freq_set)
    print('----------------------频繁3项集的规则集合----------------------------')
    # 频繁3项集的所有规则
    strong_rules = []
    for i in range(0, len(freq_set3)):
        for j in c_rule(freq_set3[i, :]):
            rule = np.array(j)
            conf = compute_conf(rule, D)
            if conf >= min_conf:
                strong_rules.append(rule)
            print('规则：', rule, '可信度：', conf)
    print('----------------------------强规则集合----------------------------')
    print(np.array(strong_rules))

if __name__ == '__main__':
    data = np.array([[1,1,0,0,1],[0,1,0,1,0],[0,1,1,0,0],[1,1,0,1,0],[1,0,1,0,0],[0,1,1,0,0],[1,0,1,0,0],[1,1,1,0,1],[1,1,1,0,0]])
    apriori(data,0.2,0.6)