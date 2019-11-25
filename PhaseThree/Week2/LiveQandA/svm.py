import matplotlib.pyplot as plt
import numpy as np

from PhaseThree.Week2.LiveQandA import utils


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    一个简化版的SMO算法实现
    :param dataMatIn: 数据输入
    :param classLabels: 数据对应的分类标签
    :param C: 松弛变量
    :param toler: 容错率
    :param maxIter: 最大迭代次数
    :return:
    """
    # 首先将输入数据转化成numpy的mat矩阵存储，维度为（100，2）
    X = np.mat(dataMatIn)
    # 将输入标签转化成numpy的mat矩阵存储，并且转置成(100,1)
    Y = np.mat(classLabels).transpose()
    b = 0
    # 获取dataMatrix的维度，m=100，n=2
    m, n = np.shape(X)
    # 对于每一组数据都初始化一个拉格朗日乘子
    lambdas = np.mat(np.zeros((m, 1)))
    # 初始化迭代
    item_num = 0
    while item_num < maxIter:
        lambdaPairChanged = 0
        for i in range(m):
            # 步骤一：计算误差Ei
            # fxi = w^Txi+b
            fxi = float(np.multiply(lambdas, Y).T * (X * X[i, :].T)) + b
            # 误差项计算
            Ei = fxi - float(Y[i])
            # 优化alpha，设定一定的容错率
            if (Y[i] * Ei < -toler and lambdas[i] < C) or (Y[i] * Ei > toler and lambdas[i] > 0):
                # 随机选择另一个lambdaJ组合一起进行优化
                j = utils.selectJrand(i, m)
                # 计算lambdaJ对应的误差Ej
                fxj = float(np.multiply(lambdas, Y).T * (X * X[j, :].T)) + b
                Ej = fxj - float(Y[j])
                # 保存更新前的lambda值
                lambdaIold = lambdas[i].copy()
                lambdaJold = lambdas[j].copy()
                # 步骤2：计算上下界H和L
                if Y[i] != Y[j]:
                    L = max(0, lambdas[j] - lambdas[i])
                    H = min(C, C + lambdas[j] - lambdas[i])
                else:
                    L = max(0, lambdas[j] + lambdas[i] - C)
                    H = min(C, lambdas[j] + lambdas[i])
                if L == H:
                    print("L == H")
                    continue
                # 步骤3：计算eta
                eta = X[i, :] * X[i, :].T + X[j, :] * X[j, :].T - 2.0 * X[i, :] * X[j, :].T
                if eta <= 0:
                    print("eta <= 0")
                    continue
                # 步骤4：更新lambdaJ
                lambdas[j] += Y[j] * (Ei - Ej) / eta
                # 步骤5：修剪lambdaJ
                lambdas[j] = utils.clipLambda(lambdas[j], H, L)
                if abs(lambdas[j] - lambdaJold) < 0.00001:
                    print("alphaJ 变化太小了")
                    continue
                # 步骤6：更新lambdaI
                lambdas[i] += Y[j] * Y[i] * (lambdaJold - lambdas[j])
                # 步骤7：更新b1和b2
                b1 = b - Ei - Y[i] * (lambdas[i] - lambdaIold) * X[i, :] * X[i, :].T - \
                     Y[j] * (lambdas[j] - lambdaJold) * X[j, :] * X[i, :].T
                b2 = b - Ej - Y[i] * (lambdas[i] - lambdaIold) * X[i, :] * X[j, :].T - \
                     Y[j] * (lambdas[j] - lambdaJold) * X[j, :] * X[j, :].T

                # 步骤8：根据b1和b2更新b
                if 0 < lambdas[i] < C:
                    b = b1
                elif 0 < lambdas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                lambdaPairChanged += 1
                print("第%d次迭代 样本：%d， alpha优化次数：%d" % (item_num, i, lambdaPairChanged))
        if lambdaPairChanged == 0:
            item_num += 1
        else:
            item_num = 0
        print("迭代次数：%d" % item_num)
    return b, lambdas


def get_w(dataMat, labelMat, lambdas):
    """
    :param dataMat: 数据矩阵
    :param labelMat: 数据标签
    :param lambdas: lambdas
    :return: 直线法向量
    """
    lambdas, dataMat, labelMat = np.array(lambdas), np.array(dataMat), np.array(labelMat)
    # 我们不知道labelMat的shape属性是多少，
    # 但是想让labelMat变成只有一列，行数不知道多少，
    # 通过labelMat.reshape(1, -1)，Numpy自动计算出有100行，
    # 新的数组shape属性为(100, 1)
    # np.tile(labelMat.reshape(1, -1).T, (1, 2))将labelMat扩展为两列(将第1列复制得到第2列)
    # dot()函数是矩阵乘，而*则表示逐个元素相乘
    # w = sum(alpha_i * yi * xi)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, lambdas)
    return w.tolist()


def showClassifer(dataMat, w, b):
    # 正样本
    data_plus = []
    # 负样本
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)
    # 正样本散点图（scatter）
    # transpose转置
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    # 负样本散点图（scatter）
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    # enumerate在字典上是枚举、列举的意思
    for i, alpha in enumerate(lambdas):
        # 支持向量机的点
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolors='red')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = utils.loadDataSet("testSet.txt")
    b, lambdas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, lambdas)
    showClassifer(dataMat, w, b)
