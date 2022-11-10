import numpy as np
import matplotlib.pyplot as plt

# 输入数据集
def createDataset():
    # dataset = [[0,0],[1,0],[0,1],[1,1],[2,1],[1,2],[2,2],[3,2],[6,6],[7,6],[8,6],[6,7],[7,7],[8,7],[9,7],[7,8],[8,8],[9,8],[8,9],[9,9]]
    dataset = [[0,0],[1,1],[2,1],[4,3],[5,3],[5,4],[6,5],[1,4],[1,5],[1,6]] # 纸质作业的测试数据
    return dataset

# 计算欧氏距离
def distance(x,y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

# 寻找质心
def classify(dataset,centroids,k):
    print('初始聚类中心：',centroids)
    classify_list = []  # 分类点与对应质心的列表
    for i in range(k):
        classify_list.append([])
    for i in range(len(dataset)):
        dist_list = [] # 某个点与n个质心之间的距离列表
        for j in range(k):
            dist = distance(centroids[j],dataset[i]) # 求点与质心之间的距离
            dist_list.append(dist)
        minDist = np.min(dist_list) # 求某个点与质心的最短距离
        classify_list[dist_list.index(minDist)].append(i) # 在对应的分类里添加点
    print('本轮分类：',classify_list)
    # 根据新的类建立聚类中心
    new_centroids = []
    for i in range(k):
        x_list = []
        y_list = []
        for j in classify_list[i]:
            x_list.append(dataset[j][0])
            y_list.append(dataset[j][1])
        new_centroids.append([np.average(np.array(x_list)),np.average(np.array(y_list))])
    print('更新后的聚类中心：',new_centroids)
    # 绘图
    paint_scatter(dataset,centroids)
    # 递归出口，如果新旧两个聚类中心相同，则停止计算，否则继续计算
    if centroids == new_centroids:
        print('最终的聚类中心：', new_centroids)
        return
    else:
        classify(dataset, new_centroids, k)

# 绘图函数
def paint_scatter(dataset,centroids):
    plt.scatter([i[0] for i in dataset], [i[1] for i in dataset]) # 其他点
    plt.scatter([i[0] for i in centroids], [i[1] for i in centroids],c='r') # 聚类点
    plt.show()

# k-means主函数
def k_means(dataset,k):
    centroids = []  # 质心列表
    for i in range(k): # 初始化质心列表
        centroids.append(dataset[i])
    classify(dataset, centroids, k)

if __name__=='__main__':
    dataset = createDataset()
    k = 3
    k_means(dataset, k)