# coding:utf-8
__author__ = 'devin'

'''
《机器学习实战》第二章 k-近邻算法
1. calculate the distance from current point to the each point which have classified
2. sort the distance in increasing order
3. select k-closest points
4. count the frequency of each category in the k points
5. return the point which have highest frequency
'''
from numpy import *
import operator


def createData():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # rows
    # calculate distance
    # construct a matrix which have same rows an columns as dataSate,and the each row is a copy of inX
    inXMat = tile(inX, (dataSetSize, 1))
    diffMat = inXMat - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == "__main__":
    group, labels = createData()
    print classify([0, 0], group, labels, 3)




