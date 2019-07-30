import matplotlib.pylab as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs   #测试数据

X, y = make_blobs(n_samples=300, centers=6, random_state=0, cluster_std=1)
#plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='rainbow')
#plt.show()

#训练集测试集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)   #stratify,分层抽样，解决样本不平均的问题

#模型训练
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)

#模型测试
print(metrics.confusion_matrix(y_test, y_pred))   #纵轴坐标为实际的y，横轴坐标为预测的y
print(metrics.classification_report(y_test, y_pred))   #准确率，召回率，f1-score准确率召回率平均效果

#画出分类区域
from mlxtend.plotting import plot_decision_regions   #画分类图
plot_decision_regions(X_train, y_train, clf=gbc)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper right')   #显示图例
plt.tight_layout()
plt.show()
