from sklearn.decomposition import PCA
import pymysql
import numpy as npy
import pandas as pd

data=pd.DataFrame([[11.010,11.059,13.240,6.458,5.830,7.334,3.994,12.008,12.914,5.547,4.569,6.438,6.365,6.054,3.965,4.812,9.150,4.428,5.961,5.844,10.070,7.086,10.869,5.728,5.387,3.595,5.942],
                   [2.4382,3.6484,5.2456,2.9337,4.9969,4.4311,1.8205,1.0158,3.8599,3.2459,0.3816,2.1628,1.3388,2.1659,0.8886,1.1620,1.6504,1.7396,9.0269,0.9641,8.7937,4.4666,12.6821,6.8689,2.5789,2.7369,4.7758]],
                   index=["danning","baililuchun"])
data=data.T

###主成分分析###
pca1=PCA()
pca1.fit(data)
#求模型中各个特征向量
charact=pca1.components_
print(charact)
#各成分贡献率   一般降维选取前85%的成分
rate=pca1.explained_variance_ratio_
print(rate)

###降维###
pca2=PCA(1)   #期望降的纬数
pca2.fit(data)
reduct=pca2.transform(data)   #降维函数
print(reduct)
recover=pca2.inverse_transform(reduct)   #降维的逆向操作
print(recover)

#选取数据类型为连续变量的特征
data_select = pd.read_csv('C:/Users/YZY/Desktop/kaggle/test.csv')
numeric_feats = data_select.dtypes[[dtype in ('float64','int64') for dtype in data_select.dtypes]].index
