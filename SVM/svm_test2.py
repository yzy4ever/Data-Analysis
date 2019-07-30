import matplotlib.pylab as plt
from sklearn import svm
from sklearn.datasets import make_blobs   #测试数据
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np


X, y = make_blobs(n_samples=300, centers=8, random_state=2, cluster_std=1)
#plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='rainbow')
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)   #stratify,分层抽样，解决样本不平均的问题

def svm_c(x_train, x_test, y_train, y_test):
    svc = svm.SVC(kernel='rbf',   #核函数：rbf linear poly
                  class_weight='balanced',   #‘balance’该参数表示给每个类别分别设置不同的惩罚参数C
                  )
    c_range = np.logspace(-5, 15, 11, base=2)   #2**-5 到 2**14 生成是一个等比数列
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)   #n_jobs=-1使CPU多核运算
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    print('精度为%s' % score)
    # 画出分类区域
    from mlxtend.plotting import plot_decision_regions  # 画分类图
    plot_decision_regions(X_train, y_train, clf=clf)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper right')  # 显示图例
    plt.tight_layout()
    plt.show()


svm_c(X_train, X_test, y_train, y_test)
