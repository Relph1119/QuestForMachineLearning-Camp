import random


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName, "r", encoding="utf-8")
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
    :param i: lambda 下标
    :param m: lambda的总数
    :return: 随机选择的下标
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipLambda(lambdaJ, H, L):
    """
    对lambdaJ做限制，当大于H或者小于L时，进行调整
    """
    if lambdaJ > H:
        lambdaJ = H
    if L > lambdaJ:
        lambdaJ = L
    return lambdaJ

