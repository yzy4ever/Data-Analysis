import numpy as np
import pandas as pd

data = pd.read_csv('C:/Users/YZY/Desktop/kaggle/train.csv')
final = pd.read_csv('C:/Users/YZY/Desktop/kaggle/test.csv')
final['SalePrice']=0

#去除空值多的特征
summary = data.isnull().sum()
delete_list = summary>=730   #如果该列缺失值超过3个就舍弃这一列
summary[delete_list]   #删除的列的columns和该列缺失值个数
data_select = data.drop(data.columns[delete_list],axis=1)
final_select = final.drop(final.columns[delete_list],axis=1)


# 缺失值如果是数值类型填充平均数，如果是离散类型填充众数
func = lambda x:x.fillna(x.mean()) if x.values.dtype != object else x.fillna(x.mode()[0])
data_select = data_select.apply(func , axis=0)
final_select = final_select.apply(func , axis=0)

#将离散变量变为哑变量
data_select = pd.get_dummies(data_select)
final_select = pd.get_dummies(final_select)

#求交集
intersection = set(data_select.keys()).intersection(set(final_select.keys()))
data_select = data_select[list(intersection)]
final_select = final_select[list(intersection)]

#特征选择
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

train,test = train_test_split(data_select , test_size=0.1)

#设置x,y
train_y = train.pop('SalePrice')   #MEDV的列表示房价
train_x = train

test_y = test.pop('SalePrice')
test_x = test


#隔点搜索
param_range = np.logspace(-6, 6, num = 40)   #从 10e-6 到 10e-1 生成20个数
#计算每个alpha算出的得分

Scores = []
for par in param_range:
    clf = Lasso(alpha = par)
    Scores.append(cross_val_score(clf, train_x, train_y, scoring='r2', cv=5))


#Lasso回归
alpha = np.linspace(-6, 6, 40)[np.array(Scores).mean(axis = 1).argmax()]
clf = Lasso(alpha=alpha)
clf.fit(train_x, train_y)
index = clf.coef_ !=0   #筛选特征,非0的clf.coef_是被筛选好的特征
index = list(train_x.columns[index])   #index为选择好的特征
index.remove('Id')

from sklearn.metrics import r2_score
lr = Lasso(alpha =0.001, random_state=1)
lr.fit(train_x[index], train_y)
y_pred = lr.predict(final_select[index])

output = pd.DataFrame({'Id':final_select.Id,'SalePrice':y_pred})
output.to_csv('submission.csv', index=False)