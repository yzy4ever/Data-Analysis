import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取数据
train = pd.read_csv(r'C:\Users\YZY\Desktop\kaggle\train.csv')
test = pd.read_csv(r'C:\Users\YZY\Desktop\kaggle\test.csv')

#移除ID信息
train_ID = train['Id']
test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)


#查看价格数据的主要分布
#train['SalePrice'].plot.kde()

#使得价格数据趋近于正态分布
train['SalePrice'] = np.log(train['SalePrice'])
#train['SalePrice'].plot.kde()


#排除评级过低但价格过高，评级过高价格过低 的不符合事实的错误数据
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)


#将训练集的特征和结果分开
train_y = train.pop('SalePrice')
train_x = train
test_x = test

#将训练集和测试集合并 便于控制填充和特征选择
all_x = pd.concat([train_x,test_x]).reset_index(drop=True)


#一些以数字表示的离散特征被默认设置为int型，将这些特征设为string型
all_x['MSSubClass'] = all_x['MSSubClass'].apply(str)
all_x['YrSold'] = all_x['YrSold'].apply(str)
all_x['MoSold'] = all_x['MoSold'].apply(str)

#缺失值比率
missing_rate = all_x.isnull().mean(axis=0).sort_values()
#pd.DataFrame(missing_rate).plot(kind='bar', stacked=True)

#去除空值过多的特征
delete_list = missing_rate[-6:].index.tolist()
all_x = all_x.drop(delete_list,axis=1)

# 缺失值如果是数值类型填充平均数，如果是离散类型填充众数
func = lambda x:x.fillna(x.mean()) if x.values.dtype != object else x.fillna(x.mode()[0])
all_x = all_x.apply(func , axis=0)

#获取所有数值型特征，存储在numerics2
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in all_x.columns:
    if all_x[i].dtype in numeric_dtypes:
        numerics2.append(i)

#在numerics2中寻找偏态分布特征
from scipy.stats import skew
skew_features = all_x[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

skewness = pd.DataFrame({'Skew' :high_skew})

#使用BOX-COX方法使偏态分布特征正规化
from scipy.special import boxcox1p, inv_boxcox1p
from scipy.stats import boxcox_normmax   #计算最佳BOX-COX转换系数lmbda
for i in skew_index:
    all_x[i] = boxcox1p(all_x[i], boxcox_normmax(all_x[i] + 1))   #使用inv_coxbox(y,lmbda)可以把boxcox处理后的值还原

#根据已有的线性特征，创建一些新非线性的特征

#离散变量转哑变量
all_x = pd.get_dummies(all_x).reset_index(drop=True)

#分离测试集和验证集
train_x = all_x.iloc[:len(train_y), :]
test_x = all_x.iloc[len(train_y):, :]

#训练模型
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, LassoCV, LassoLarsCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import cross_val_score

#或者不看图直接用clf = LassoLarsCV(cv=5).fit(train_x, train_y)
clf = RidgeCV(cv=5).fit(train_x, train_y)
#clf = LassoLarsCV(cv=5).fit(train_x, train_y)
#clf = ElasticNetCV(cv=5).fit(train_x, train_y)
y_pred = clf.predict(test_x)

import math
output = pd.DataFrame({'Id':test_ID,'SalePrice':math.e**y_pred})
output.to_csv('submission.csv', index=False)

