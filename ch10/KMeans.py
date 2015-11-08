# coding: utf-8
__author__ = 'devin'
from numpy import *
import pylab as pl

def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as f:
        for line in f.readlines():
            curLine = line.strip().split("\t")
            fltLine = map(float, curLine)  # 转换为float
            dataMat.append(fltLine)
    return dataMat

# 计算欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))


# 随机取k个中心
def randCent(dataSet, k):
    n = shape(dataSet)[1]  # 列数
    centroids = mat(zeros((k, n))) # k行n列的矩阵 也就是取k个n维向量
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)  # 生成j列向量

    return centroids

# k-means算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    创建k个点作为起始质心(随机选择)
    当任意一个点的簇分配结果改变时:
        对数据集中的每个数据点
             对每个质心
                  计算质心到数据点之间的距离
             讲数据点分配到距其最近的簇
        对每一个簇,计算簇中所有点的均值幷讲均值作为质心
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # 第一列记录最近簇的索引,第二咧是距离
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 更新质心的位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def binKMeans(dataSet, k, distMeas=distEclud):
    """
    :param dataSet:
    :param k:
    :param distMeas:
    :return:
    选择一个初始的簇中心(取均值),加入簇中心列表
    计算每个数据点到簇中心的距离
    当簇的个数小于指定的k时
          对已经存在的每个簇进行2-均值划分,并计算其划分后总的SSE,找到最小的划分簇
          增加一个簇幷更新数据点的簇聚类
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # 创建一个初始簇, 取每一维的平均值
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # 记录有几个簇
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
    #     # 找到对所有簇中单个簇进行2-means可以是所有簇的sse最小的簇
        for i in range(len(centList)):
            # 属于第i簇的数据
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # print ptsInCurrCluster
            # 对第i簇进行2-means
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 第i簇2-means的sse值
            sseSplit = sum(splitClustAss[:, 1])
            # 不属于第i簇的sse值
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i), 1])
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNotSplit + sseSplit
        # 更新簇的分配结果
        #新增的簇编号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        #另一个编号改为被分割的簇的编号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  #
        # 更新被分割的的编号的簇的质心
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        # 添加新的簇质心
        centList.append(bestNewCents[1, :].tolist()[0])
        # 更新原来的cluster assment
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment


if __name__ == "__main__":
    dataMat = mat(loadDataSet('testSet2.txt'))
    # print mean(dataMat, axis=0).tolist()[0]
    centroids, clusterAssment = binKMeans(dataMat, 4)
    # centroids, clusterAssment = kMeans(dataMat, 4)
    pl.plot(centroids[:, 0], centroids[:, 1], 'ro')
    pl.plot(dataMat[:, 0], dataMat[:, 1], 'bo')
    pl.show()









