#Ridge二阶正则化 Lasso一阶正则化 ElasticNet是两种的混合
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

'''
def re_cv(model):
    rmse = cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5)
    return rmse
'''

#读取房价数据
boston=pd.read_csv("C:/Users/YZY/Anaconda3/Lib/site-packages/sklearn/datasets/data/boston_house_prices.csv",skiprows=1)   #skiprows忽略第1行


#自动的将数据划分为训练集和测试集
train,test = train_test_split(boston , test_size=0.1)   #约定测试数据占比0.1   #stratify=y,分层抽样，解决样本不平均的问题

#设置x,y
train_y = train.pop('MEDV')   #MEDV的列表示房价
train_x = train

test_y = test.pop('MEDV')
test_x = test

#直接用线性拟合
clf = LinearRegression()
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
print(r2_score(y_true=test_y, y_pred=y_pred))   #r方交叉验证的得分

#使用Lasso回归，Lasso回归的关键是确定alpha
clf = Lasso(alpha=0.0)
clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
print(r2_score(y_true=test_y, y_pred=y_pred))   #在alpha=0时与线性拟合基本一致

#为确定alpha，采用隔点搜索
#用logspace先产生一组非常微小的数
param_range = np.logspace(-6, -1, num = 20)   #从 10e-6 到 10e-1 生成20个等比数列

#计算每个alpha算出的得分
Scores = []
for par in param_range:
    clf = Lasso(alpha = par)
    Scores.append(cross_val_score(clf, train_x, train_y, scoring='r2', cv=5))

plt.xlabel('alpha')
plt.ylabel('r2')
plt.plot(np.linspace(-6, -1, 20), np.array(Scores).mean(axis = 1))
plt.show()   #取r2最大值的alpha为-2.3，那么alpha为10**-2.3

#Lasso回归
clf = Lasso(alpha=10**-2.3)
clf.fit(train_x, train_y)
index = clf.coef_ !=0   #筛选特征,非0的clf.coef_是被筛选好的特征
index = list(test_x.columns[index])   #index为选择好的特征

#将选择好的特征进行训练
lr = LinearRegression()
lr.fit(train_x[index], train_y)
y_pred = lr.predict(test_x[index])
r2_score(y_true=test_y, y_pred=y_pred)   #最终评估得分






