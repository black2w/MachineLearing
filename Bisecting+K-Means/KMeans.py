#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np  # 导入NumPy
import pandas as pd  # 导入pandas
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv')
print(data.head())  # 显示一些数据
# 只针对两个特征进行聚类，以方便二维的展示

print(data.dtypes)

print(data.isnull().any())

'''
data['Income'].plot()
plt.title('Income of customers')
plt.xlabel('ID')
plt.ylabel('Income')
plt.show()

data['Spending'].plot()
plt.title('Spending of customers')
plt.xlabel('ID')
plt.ylabel('Spending')
plt.show()

df_tmp1 = data[['Age', 'Income', 'Spending']]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df_tmp1.corr(), cmap="YlGnBu", annot=True)
plt.show()
'''

X = data.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans  # 导入聚类模型

cost = []  # 初始化损失(距离)值
for i in range(1, 11):  # 尝试不同的K值
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    cost.append(kmeans.inertia_)  # inertia_ 是我们选择的方法，其作用相当于损失函数

import matplotlib.pyplot as plt  # 导入matplotlib库
import seaborn as sns  # 导入seaborn库

# 绘制ELBOW（手肘）图找到最佳K值
plt.plot(range(1, 11), cost)
plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('Cost')
plt.show()

kmeansmodel = KMeans(n_clusters=4, init='k-means++')  # 选择4作为聚类个数
y_kmeans = kmeansmodel.fit_predict(X)  # 进行聚类的拟合和分类

# 下面把分好的聚类可视化
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
            s=100, c='cyan', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
            s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],
            s=100, c='red', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
#
# data_tmp = data[['Income', 'Spending']]
#
# # 简单打印结果
# r1 = pd.Series(kmeansmodel.labels_).value_counts()  # 统计各个类别的数目
# r2 = pd.DataFrame(kmeansmodel.cluster_centers_)  # 找出聚类中心
# r = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
# r.columns = list(data_tmp.columns) + ['类别数目']  # 重命名表头
# print(r)
#
# # 详细输出原始数据及其类别
# r = pd.concat([data_tmp, pd.Series(kmeansmodel.labels_, index=data_tmp.index)], axis=1)  # 详细输出每个样本对应的类别
# r.columns = list(data_tmp.columns) + ['聚类类别']  # 重命名表头
# r.to_excel('result_output.xlsx')  # 保存结果
#
# k = 4
#
#
# def density_plot(data):  # 自定义作图函数
#     import matplotlib.pyplot as plt
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#     p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
#     [p[i].set_ylabel(u'密度') for i in range(2)]
#     plt.legend()
#     return plt
#
#
# pic_output = './pd'  # 概率密度图文件名前缀
#
# for i in range(k):
#     print(data_tmp[r[u'聚类类别'] == i])
#     density_plot(data_tmp[r[u'聚类类别'] == i]).savefig(u'%s%s.png' % (pic_output, i))


