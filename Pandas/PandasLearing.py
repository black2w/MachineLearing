import pandas as pd # 这里的pandas 是一个读取表格的库 类比的话像 js中的 xlsx库

df = pd.read_excel('/Users/black2w/Documents/GitHub/MachineLearing/Pandas/food.xlsx')
print("#########打印默认前部数据#############")
print(df.head())
print("##########打印数据前10行############")
print(df.head(10))
print("##########表格有多少列？############")
print(df.columns)
print("##########数据索引############")
print(df.index)
# print("#########能量最高食品品############")
# c = df[['食物名','能量(千卡）']].groupby(['能量(千卡）'], as_index=False)
# c.sort(['能量(千卡）'],ascending=False,inplace=True)
# print(c.head)
print("#########能量最高食品品############")