import numpy as np

# 创建数据集
def dataset():
    # data[:,0] 0表示女，1表示男，data[:,1] 表示身高
    data = np.array([[0,1.6],[1,2],[0,1.9],[0,1.88],[0,1.7],[1,1.85],[0,1.6],[1,1.7],[1,2.2],[1,2.1],[0,1.8],[1,1.95],[0,1.9],[0,1.8],[0,1.75]])
    labels = np.array(['矮','高','中等','中等','矮','中等','矮','矮','高','高','中等','中等','中等','中等','中等'])
    return data, labels

# knn函数
def knn(test_data, train_data, train_labels, k):
    # 计算欧式距离
    distance = np.zeros(train_data.shape[0])
    for i in range(train_data.shape[0]):
        distance[i] = (train_data[i][0] - test_data[0]) ** 2 + (train_data[i][1] - test_data[1]) ** 2
        distance[i] = np.sqrt(np.abs(distance[i]))
    # 返回按距离排序的索引
    index = distance.argsort()
    # 标签集
    labels = {'高': 0, '中等': 0, '矮': 0}
    for i in range(k):
        if(train_labels[index[i]] == '高'):
            labels['高'] += 1
        elif(train_labels[index[i]] == '中等'):
            labels['中等'] += 1
        else:
            labels['矮'] += 1
    return max(labels, key=labels.get)

# 主函数，计算最终准确率（性能结果）
if __name__=='__main__':
    data, labels = dataset()
    k = 3 # 连续n个样本
    test_times = int(len(data)/k) # n个测试集合
    res = np.ones(test_times) # 准确率结构集
    for i in range(test_times):
        # 将测试数据集和训练数据集进行分类
        test_data = data[i*k:(i+1)*k]
        test_label = labels[i*k:(i+1)*k]
        train_data = np.concatenate((data[0:i*k],data[(i+1)*k:len(data)]))
        train_labels = np.concatenate((labels[0:i*k],labels[(i+1)*k:len(data)]))
        for j in range(k):
            res_i = knn(test_data[j],train_data,train_labels,k)
            if(res_i != test_label[j]): # 如果错误,准确率每次减去1/k
                res[i] = res[i] - 1/k
    print('最终准确率为：',np.average(res))
