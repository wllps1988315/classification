# -*- coding:utf-8 -*-
'''
DBSCAN聚类算法概述：
DBSCAN属于密度聚类算法，把类定义为密度相连对象的最大集合，通过在样本空间中不断搜索最大集合完成聚类。
DBSCAN能够在带有噪点的样本空间中发现任意形状的聚类并排除噪点。
DBSCAN算法不需要预先指定聚类数量，但对用户设定的参数非常敏感。
当空间聚类的密度不均匀、聚类间距差相差很大时，聚类质量较差。

DBSCAN算法基本概念：
核心对象：如果给定对象的半径eps邻域内样本数量超过阈值min_samples，则称为核心对象。
边界对象：在半径eps内点的数量小于min_samples，但是落在核心点的邻域内。
噪声对象：既不是核心对象也不是边界对象的样本。
直接密度可达：如果对象q在核心对象p的eps邻域内，则称q从p出发是直接密度可达的。
密度可达：集合中的对象链p1、p2、p3、...、pn，如果每个对象pi+1从pi出发都是直接密度可达的，则称pn从p1出发是密度可达的。
密度相连：集合中如果存在对象o使得对象p和q从o出发都是密度可达的，则称对象p和q是互相密度相连的。

DBSCAN聚类算法工作过程：
1）定义邻域半径eps和样本数量阈值min_samples。
2）从样本空间中抽取一个尚未访问过的样本p。
3）如果样本p是核心对象，进入第4）步；否则返回第2）步。
4）找出样本p出发的所有密度可达对象，构成一个聚类Cp（该聚类的边界对象都是非核心对象），并标记这些对象为已访问。
5）如果全部样本都已访问，算法结束；否则返回第2）步。
'''
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def generateData():
    '''
    生成数据
    :return:
    '''
    def get(start,end):
        return [randrange(start,end) for _ in range(50)]

    x1 = get(0,40)
    x2 = get(70, 100)
    y1 = get(0, 30)
    y2 = get(40, 70)

    data = list(zip(x1,y1)) + list(zip(x1,y2)) +\
            list(zip(x2,y1)) + list(zip(x2,y2))

    return np.array(data)

def main(data,eps=0.3,min_samples=10):
    #聚类
    db = DBSCAN(eps=eps,min_samples=min_samples).fit(data)
    #标记核心对象对应下表为True
    coreSamplesMask = np.zeros_like(db.labels_,dtype=bool)
    coreSamplesMask[db.core_sample_indices_] = True

    #聚类标签(数组,表示每个样本所属聚类)和所有聚类数量,标签-1对应的样本表示噪点
    clusterLabels = db.labels_
    uniqueClusterLabels = set(clusterLabels)
    nClusters = len(uniqueClusterLabels) - (-1 in clusterLabels)

    #绘制聚类结果
    colors = ['red','green','black','gray','#ff00ff','#ffff00']
    markers =['v','^','o','*','x','h','d']

    for i,cluster in enumerate(uniqueClusterLabels):
        print('聚类标签为{}的数据'.format(cluster).center(40,'='))
        #clusterIndex是个True/False数组,其中True表示对应样本为核心对象
        clusterIndex = (clusterLabels == cluster)

        #绘制核心对象
        coreSamples = data[clusterIndex&coreSamplesMask]
        print('核心对象'.ljust(30,'*'))
        print(coreSamples)
        plt.scatter(coreSamples[:,0],coreSamples[:,1],c=colors[i],marker=markers[i],s=80)

        # 绘制核心对象
        noiseSamples = data[clusterIndex&~coreSamplesMask]
        print('非核心对象'.ljust(30,'*'))
        print(noiseSamples)
        plt.scatter(noiseSamples[:,0],noiseSamples[:,1],c=colors[i],marker=markers[i],s=26)

data = generateData()
main(data,10,15)
main(data,10,10)
main(data,10,30)