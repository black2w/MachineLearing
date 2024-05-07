import pandas as pd
import  numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import psutil

#读取数据
from sklearn.cluster import KMeans

data = pd.read_csv('/Users/black2w/Documents/GitHub/MachineLearing/Bisecting+K-Means/data.csv')
print("*********打印部分数据*********")
print(data)
print("*********打印数据类型*********")
print(data.dtypes)
print("*********数据缺失值查看*********")
print(data.isnull().any())


#图表展示
fig = plt.figure()
# fig.canvas.mpl_connect('close_event', on_close)

#数据图表展示ID-收入
plt.subplot(2, 3, 1)
data['Income'].plot()
plt.title('收入分布')
plt.xlabel('用户ID')
plt.ylabel('收入')
# plt.show

##数据图表展示ID-Spending
plt.subplot(2, 3, 2)
data['Spending'].plot()
plt.title('消费率分布')
plt.xlabel('用户ID')
plt.ylabel('消费率')
# plt.show()

plt.subplot(2, 3, 3)
df_tmp1 = data[['Age', 'Income', 'Spending']]
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df_tmp1.corr(), cmap="YlGnBu", annot=True)
plt.title('年龄、收入、消费率热力图')
plt.show()

# 确定聚类的簇数 \( k \) 是聚类分析中的一个重要问题，通常需要结合数据集的特点和应用背景来进行选择。以下是一些常见的确定 \( k \) 的方法：
#
# 1. **肘部法（Elbow Method）**：这是一种直观的方法，通过观察聚类误差（如 SSE）随着 \( k \) 值增加的变化情况来确定最佳的 \( k \) 值。在绘制聚类误差随 \( k \) 值变化的曲线时，通常会出现一个肘部，即曲线开始急剧下降后趋于平缓的点，这个点对应的 \( k \) 值可以作为最佳的聚类数。
#
# 2. **轮廓系数法（Silhouette Score）**：轮廓系数是一种衡量聚类结果质量的指标，它考虑了簇内的紧密度和簇间的分离度。通过计算不同 \( k \) 值下的轮廓系数，可以选择使轮廓系数最大的 \( k \) 值作为最佳的聚类数。
#
# 3. **业务需求和先验知识**：有时候，根据业务需求和先验知识来确定 \( k \) 值可能更为合适。例如，如果我们知道数据集中代表不同类别的样本数量，那么可以选择 \( k \) 为这个数量。
#
# 4. **可视化方法**：通过可视化聚类结果，观察不同 \( k \) 值下的簇的分布和特征，从而帮助确定最佳的 \( k \) 值。
#
# 5. **交叉验证**：使用交叉验证等模型评估方法，根据模型的泛化性能来选择最佳的 \( k \) 值。
#
# 在实践中，通常会结合多种方法来确定最佳的 \( k \) 值，以确保得到合理且稳健的聚类结果。


#使用手肘图法找到最佳K值
#取第三列、第四列数据，年龄、收入
X = data.iloc[:, [3, 4]].values
np.set_printoptions(suppress=True)
print(X)  # 显示一些数据

# 定义可能的 k 值范围
k_values = range(1, 11)
# 计算每个 k 值对应的 SSE
cost = []  # 初始化损失(距离)值
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X)
    cost.append(kmeans.inertia_)

# 绘制 SSE 随 k 变化的曲线

#画出/
plt.subplot(2, 3, 4)
plt.plot(k_values, cost, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()


def bisecting_kmeans(data, k, max_iters=4):
    # 初始时将所有数据视为一个簇
    clusters = [data]
    while len(clusters) < k:
        # 选择要分裂的簇
        cluster_to_split = None
        max_sse_reduction = -np.inf
        for i, cluster in enumerate(clusters):
            cluster_sse = np.sum(np.square(cluster - np.mean(cluster, axis=0)))
            if cluster_sse > max_sse_reduction:
                max_sse_reduction = cluster_sse
                cluster_to_split = i

        # 对选择的簇进行 K-Means 聚类
        data_to_split = clusters.pop(cluster_to_split)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(data_to_split)
        split_clusters = [data_to_split[kmeans.labels_ == 0], data_to_split[kmeans.labels_ == 1]]

        # 将分裂的两个簇添加到列表中
        clusters.extend(split_clusters)

    return clusters



final_clusters = bisecting_kmeans(X, 4)
# 可视化聚类结果
plt.subplot(2, 3, 6)
colors = ['red', 'blue', 'green', 'purple']
for i, cluster in enumerate(final_clusters):
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i+1}')
plt.title('Bisecting K-Means Clustering')
plt.xlabel('收入')
plt.ylabel('消费率')
plt.legend()
plt.show()



# plt.close()