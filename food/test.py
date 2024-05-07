import torch
import pandas as pd # 这里的pandas 是一个读取表格的库 类比的话像 js中的 xlsx库
import numpy as np
from sklearn.cluster import KMeans #KMeans 库
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt #制图

def savaData(filePath, data):
    '''
    用于保存输出结果到指定路径下
    :param filePath: 保存结果的目的文件路径
    :param data: 结果数据
    :return:
    '''
    file = open(filePath, 'w+', encoding='utf-8')  # 注意规定编码格式
    file.write(str(data))  # 写入结果数据
    file.close()


df = pd.read_excel('food/food.xlsx')
df1 = df.dropna()  # 删除含有数据缺失的行
data = df1.drop('食物名', axis=1, inplace=False)  # 删除'食物名'列 axis=0代表删除行,1代表删除列 inplace=False代表不改变原表 True代表改变原表
data = data.drop('序号', axis=1, inplace=False)  # 删除'序号'列
print(data.head())

# 数据标准化
z_scaler = preprocessing.StandardScaler()
data_z = z_scaler.fit_transform(data)
data_z = pd.DataFrame(data_z)


# 数据归一化
minmax_scale = preprocessing.MinMaxScaler().fit(data_z)
dataa = minmax_scale.transform(data_z)
print(pd.DataFrame(dataa).head())


K = range(1, 11) #

meanDistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dataa)
    # 计算各个点分别到k个质心的距离,取最小值作为其到所属质心的距离,并计算这些点到各自所属质心距离的平均距离
    meanDistortions.append(
        sum(
            np.min(cdist(dataa, kmeans.cluster_centers_, 'euclidean'), axis=1)
        ) / dataa.shape[0]
    )
# 绘制碎石图
plt.plot(K, meanDistortions,'bx--')
plt.xlabel('k')
plt.show()

# init='k-means++'表示用kmeans++的方法来选择初始质数 n_clusters=8表示数据聚为8类 max_iter=500表示最大的迭代次数是500
k_means=KMeans(init='k-means++',n_clusters=8,max_iter=500)
k_means.fit(dataa)
label = k_means.fit_predict(dataa)  # 聚类后的聚类标签放在label内
print(label)


data1 = df1['食物名']
data2 = data1.values
# 储存聚类结果
FoodCluster = [[], [], [], [], [],[], [], []]  # 分为8类
for i in range(len(data2)):
    FoodCluster[label[i]].append(data2[i])

resultStr = ''  # 保存分类结果
# 输出分类结果
for i in range(len(FoodCluster)):
    print(FoodCluster[i])
    # 将同分类食物用,拼接
    resultStr = resultStr + ','.join(FoodCluster[i]) + '\n'
savaData('/resultF.csv', resultStr)