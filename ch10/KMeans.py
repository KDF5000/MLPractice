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


if __name__ == "__main__":
    dataMat = mat(loadDataSet('testSet.txt'))
    # print dataMat
    centroids, clusterAssment = kMeans(dataMat, 4)
    # print centroids, clusterAssment
    pl.plot(centroids[:, 0], centroids[:, 1], 'ro')
    pl.plot(dataMat[:, 0], dataMat[:, 1], 'bo')
    pl.show()









