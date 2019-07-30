'''
数据清洗 填充空值 类型转换
'''
import numpy as np
import pandas as pd

data=pd.DataFrame([[1,2,3,None,5,6,"M"],
                   [7,8,9,None,11,12,"F"],
                   [13,None,14,15,16,17,"F"],
                   [18,19,20,None,21,22,None],
                   [23,24,25,26,27,28,"M"],
                   [None,29,30,31,32,33,"F"]],
                  columns=["A","B","C","D","E","F","G"])


#去除空值多的特征
summary = data.isnull().sum()
delete_list = summary>=3   #如果该列缺失值超过3个就舍弃这一列
summary[delete_list]   #删除的列的columns和该列缺失值个数
data_select = data.drop(data.columns[delete_list],axis=1)

# 缺失值如果是数值类型填充平均数，如果是离散类型填充众数
func = lambda x:x.fillna(x.mean()) if x.values.dtype != object else x.fillna(x.mode()[0])
data_select = data_select.apply(func , axis=0)

#将离散变量变为哑变量
data_select = pd.get_dummies(data_select)

#特征工程
#让数据分布更加接近正态分布
import matplotlib.pylab as plt
data_select["A"] = np.log(data_select.A)   #对右偏分布的数据求对数使其接近正态分布
plt.hist(data_select.A)
plt.show()
