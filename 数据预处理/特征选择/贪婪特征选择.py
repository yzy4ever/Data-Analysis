#数据分为训练集、验证集、测试集

#贪婪特征选择法就是每次加一个特征，加一个能使交叉验证结果最优的那个特征

#对训练集和验证集进行交叉验证
#交叉验证是将训练集和验证集轮换

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


#贪婪特征选择法
def feature_select(train_x, train_y, lr, features):
    Scores = []
    for col in train_x.columns.drop(features):
        mask = features.copy()
        mask.append(col)
        scores = cross_validate(lr, train_x[mask], train_y, scoring='r2',cv=5,return_train_score = True)  #交叉验证使用r方的测量方法，r方=1-偏差/总误差，r方越大误差越小
        Scores.append([col, scores['test_score'].mean(), scores['test_score'].std()])
    Scores = np.array(Scores)
    index = Scores[:,1].astype(np.float).argsort()
    return Scores[:,0][index[::-1]]

#读取房价数据
boston=pd.read_csv("C:/Users/YZY/Anaconda3/Lib/site-packages/sklearn/datasets/data/boston_house_prices.csv",skiprows=1)   #skiprows忽略第1行


#自动的将数据划分为训练集和测试集
train,test = train_test_split(boston , test_size=0.1)   #约定测试数据占比0.1

#设置x,y
train_y = train.pop('MEDV')   #MEDV的列表示房价
train_x = train

test_y = test.pop('MEDV')
test_x = test

#设定方法为线性回归法
lr = LinearRegression()

#选择特征
ordered_features = []   #存储按交叉验证得分排行初步筛选出来的特征，个数比较多，无法确定到底用前多少个特征最合适
while len(ordered_features)<10:   #假设初步要筛选出10个特征，那么就迭代10次
    feature = feature_select(train_x, train_y, lr, ordered_features)[0]
    ordered_features.append(feature)

#如何确定到底用前多少个特征？
ScoresCum = []   #计算每次迭代的得分
mask = []
for col in ordered_features:
    mask.append(col)
    scores = cross_validate(lr, train_x[mask], train_y, scoring='r2',cv=5,return_train_score = True)   #交叉验证
    ScoresCum.append([col, scores['train_score'].mean(), scores['test_score'].mean(),scores['test_score'].std()])

plt.plot(pd.DataFrame(ScoresCum)[1],color = 'b')   #训练集r方会越来越大，说明随着特征越数来越多误差越来越小
plt.plot(pd.DataFrame(ScoresCum)[2],color = 'r')   #测试集r方会先增大再减小，说明随着特征越数来越多，误差先减小后增大，特征数过多反而误差会变大，原因是过拟合
plt.xlabel('features')
plt.ylabel('R2score')
plt.ylim(0.2,1)
plt.show()   #根据拐点选择特征数


#测试
mask = ordered_features
lr.fit(train_x[mask[:6]],train_y)   #选取前六个特征进行训练
y_pred = lr.predict(test_x[mask[:6]])

#模型评估
r2_score(y_true= test_y, y_pred=y_pred)

