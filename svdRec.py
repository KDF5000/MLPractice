# coding:utf-8
__author__ = 'devin'

from numpy import *

def loadData():
    return [
        [1, 1, 1, 0, 0],
        [2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 1, 1]
    ]


def loadData2():
    return [
        [2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
        [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
        [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
        [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
        [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]
    ]

# 欧氏距离 inA, inB均为列向量
def ecludSim(inA, inB):
    return 1/(1 + linalg.norm(inA-inB))  # norm计算2范式即 (a^2 + b^2 +c^2)的平方根


# 皮尔逊距离
def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]  # 归一化到0-1之间


# 余弦距离
def cosSim(inA, inB):
    num  = float(inA.T * inB)
    denum = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num/denum)


# 预测指定用户对指定物品的评级
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]  # 列数
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # 找出同一个用户对两个item都进行了评价的用户
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        simTotal += similarity
        ratSimTotal += similarity * userRating
        print userRating

    if simTotal == 0:
        print 0
        return 0
    else:
        return ratSimTotal/simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 寻找没有评级的物品
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:  # 说明该用户对所有物品都评论过了
        return 'you rated all of the items'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))

    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


# svd 评估分数
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]  # 列数
    sim_total = 0.0
    rat_sim_total = 0.0
    U, Sigma, VT = linalg.svd(dataMat)
    sig_sum = sum(Sigma)
    sig_num = len(Sigma)
    total = 0
    for latent_size in range(sig_num):
        total += Sigma[latent_size]
        print latent_size, total/sig_sum
        if total/sig_sum > 0.9:
            break

    Sig4 = mat(eye(latent_size) * Sigma[:latent_size])
    formated_items = dataMat.T * U[:, :latent_size] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        similarity = simMeas(formated_items[item, :].T, formated_items[j, :].T)
        sim_total += similarity
        rat_sim_total += similarity * userRating
    if sim_total == 0:
        return 0
    return rat_sim_total/sim_total


if __name__ == "__main__":
    # myMat = mat(loadData())
    # print ecludSim(myMat[:, 0], myMat[:, 4])
    # print pearsSim(myMat[:, 0], myMat[:, 4])
    # print cosSim(myMat[:, 0], myMat[:, 4])
    # print myMat[:, 0] - myMat[:, 4]
    data = mat(loadData2())
    res = recommend(data, 3, estMethod=svdEst)
    print res

