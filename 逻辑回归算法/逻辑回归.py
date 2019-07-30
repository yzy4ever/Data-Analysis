import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy import ndimage
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

data=pd.read_csv("luqu.csv")
x=data.iloc[:,1:5]
y=data.iloc[:,0:1]


#####首先进行特征筛选#####
rlr = RLR() #建立随机逻辑回归模型，进行特征选择和变量筛选
rlr.fit(x,y) #训练模型
egeList=rlr.get_support() #获取筛选后的特征

print("rlr.get_support():")
print(egeList)
print('有效特征为：' + str(data.iloc[:,1:5].columns[egeList].values))
print('随机逻辑回归模型特征选择结束！！！')


x = data[data.iloc[:,1:5].columns[egeList]] #筛选好特征值，排除无效的特征值



#####确定逻辑回归参数#####

# 为确定参数C值，采用隔点搜索
# 用logspace先产生一组非常微小的数
from sklearn.model_selection import cross_val_score
lr = LR()

Pars = np.logspace(-4, 4, num=20)

Scores = []
for C in Pars:
    logreg = LR(penalty='l2', C=C)   #分别把C带入模型中训练
    Scores.append(cross_val_score(logreg,x,y,scoring='accuracy',cv=5))   #交叉验证每次的得分


plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.plot(np.linspace(-4, 4, 20), np.array(Scores).mean(axis = 1))
plt.show()   #取accuracy最大值的C为0.63，那么alpha为10**0.63

C = 10**np.linspace(-4, 4, 20)[np.array(Scores).mean(axis = 1).argmax()]   #得分要取平均数 求得最高得分对应的C值

#####进行逻辑回归#####
lr = LR(C=C) #建立逻辑回归模型
lr.fit(x, y) #用筛选后的特征进行训练
print(u'逻辑回归训练模型结束！！！')
print(u'模型的平均正确率：' + str(lr.score(x, y))) #给出模型的平均正确率


print(lr.predict(pd.DataFrame([[560,4,1],[550,2.9,3]])))   #对gre=760 gpa=3.4 rank=3 预测是否录取
print(lr.predict_proba(pd.DataFrame([[560,4,1],[550,2.9,3]])))#返回一个概率数组
#对于[550,2.9,3]，y=0（不录取）的概率为0.81073788；y=1（录取）的概率为0.18926212


#####解释因子#####
w = lr.coef_   #获得参数
df = pd.DataFrame(w[0], index=x.columns.values)
df = df.sort_values(by=0)
df.plot(kind='bar', stacked=True)   #大于0呈正相关，小于0呈负相关，趋近于0不相关
