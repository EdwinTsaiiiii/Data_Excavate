import numpy as np

# 数据集导入函数
def createDataSet():
    # 0表示雨 1表示多云 2表示晴    0表示冷 1表示适中 2表示热    0表示正常 1表示高      0表示无 1表示有       0表示N 1表示P
    dataset = np.array([[2,2,1,0,0],[2,2,1,1,0],[1,2,1,0,1],[0,1,1,0,1],[0,0,0,0,1],[0,0,0,1,0],[1,0,0,1,1],[2,1,1,0,0],[2,0,0,0,1],[0,1,0,0,1],[2,1,0,1,1],[1,1,1,1,1],[1,2,0,0,1],[0,1,1,1,0]])
    labels = np.array(['天气','气温','湿度','风','类别'])
    return dataset,labels

# 先验熵
def p_entropy(s,n):
    sn = s[np.where(s==n)]
    if len(sn) != 0:
        p = len(sn) / len(s)
        return -np.sum(p * np.log2(p))
    else:
        return 0

# 后验熵和该熵的权重
def l_entropy(s,n):
    sn = s[np.where(s[:,0]==n)] # 同一属性有n个
    p1 = len(sn) / len(s)
    u1 = sn[np.where(sn[:,1]==1)]  # 同一属性中类别为P有u1个
    pu1v = len(u1) / len(sn) # p(u1|v1)
    pu2v = 1 - pu1v # p(u2|v)
    if(pu1v == 0 or pu2v == 0):
        return np.array([0,p1])
    else:
        return np.array([-np.sum(pu1v*np.log2(pu1v)) - np.sum(pu2v*np.log2(pu2v)),p1])

# 条件熵
def c_entropy(s,m):
    condition_entropy = 0
    for i in range(0,m+1):
        condition_entropy += l_entropy(s, i)[0] * l_entropy(s, i)[1] # 后验熵乘以权重再累加
    return condition_entropy

# 创建决策树
def create_tree(dataset,labels_index,labels,my_tree):
    # print('----------------数据集------------------\n', dataset)
    if np.std(np.array(dataset[:,4])) == 0: # 递归出口：数据集类别全相同
        my_tree['class'] = dataset[0,4]
        return
    pre_entropy = p_entropy(dataset[:,4],1) + p_entropy(dataset[:,4],0) # 先验熵
    each_info = {0:0,1:0,2:0,3:0} # 缓存存互信息（信息增益）的数组
    for i in labels_index:
        s = np.array([dataset[:,i],dataset[:,4]]).T
        condition_entropy = c_entropy(s,np.max(s[:,0])) # 条件熵
        each_info[i] = pre_entropy - condition_entropy # 互信息（信息增益）
    root_index = max(each_info, key=each_info.get) # 找出第0层互信息（信息增益）最大的属性的index
    my_tree[labels[root_index]] = {}
    for j in range(0, np.max(dataset[:, root_index])+1):
        new_dataset = dataset[np.where(dataset[:,root_index] == j)] # 创建新的数据集
        new_labels_index = labels_index[np.where(labels_index[:] != root_index)] # 创建新的属性的index
        my_tree[labels[root_index]][j] = {}
        create_tree(new_dataset,new_labels_index,labels,my_tree[labels[root_index]][j]) # 递归

# 预测方法,注意递归函数无法return值，只能对值进行保存或输出
def pred_fun(my_tree,labels,test_data):
    key = list(my_tree.keys())[0] # 获取决策树的第一属性
    if key == 'class': # 递归出口
        if my_tree[key] == 1:
            print(test_data,'的预测结果为：P')
            return
        if my_tree[key] == 0:
            print(test_data,'的预测结果为：N')
            return
        return
    index = list(labels).index(key) # 获取该属性的索引
    new_my_tree = my_tree.get(key)[test_data[index]] # 寻找子树
    pred_fun(new_my_tree,labels,test_data) # 递归

if __name__=='__main__':
    dataset,labels = createDataSet()
    my_tree = {}
    create_tree(dataset,np.array([0,1,2,3]),labels,my_tree)
    print('决策树：', my_tree)
    test_data = [2,1,0,1] # 测试数据
    pred_fun(my_tree,labels,test_data) # 进行预测